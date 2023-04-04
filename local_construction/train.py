from logging import warning
import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

try:
    from nets.attention_local import set_decode_type
    from utils.log_utils import log_values
    from utils import move_to
except:
    import sys
    sys.path.insert(0, './')
    from nets.attention_local import set_decode_type
    from utils.log_utils import log_values
    from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    print('Validating...')
    cost = rollout(model, dataset, opts)
    if opts.data_distribution == 'scale':
        avg_cost = cost.view(10, -1).mean(dim=1)
        print('Validation overall avg_cost:')
        print(avg_cost)
    elif opts.data_distribution == 'unit':
        avg_cost = cost.mean()
        print('Validation overall avg_cost: {} +- {}'.format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts, cal_mean=False):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, cost2, _ = model(move_to(bat, opts.device))
            if cal_mean: 
                cost = torch.stack((cost, cost2)).mean(dim=0)
            else:
                cost, _ = torch.stack((cost, cost2)).min(dim=0)
        return cost.data.cpu() 

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts, train_dataset=None):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Use the input dataset or generate new training data for each epoch
    if train_dataset is not None:
        training_dataset = baseline.wrap_dataset(train_dataset)
        training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0, shuffle=True)
    else:
        training_dataset = baseline.wrap_dataset(problem.make_dataset(
            size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
        training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1, shuffle=False)
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    avg_reward = validate(model, val_dataset, opts)
    # if not opts.no_tensorboard:
    #     tb_logger.log_value('val_avg_reward', avg_reward, step)
    
    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler[0].step()
    lr_scheduler[1].step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    # bl_val shape (batch_size, ) for rollout
    
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    # cost, log_likelihood, entropies = model(x)
    cost, log_likelihood, cost2, log_likelihood2 = model(x)
    
    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss with Scaled Entropy Reward
    # reinforce_loss = ((cost - bl_val - opts.alp*entropies.detach()) * log_likelihood).mean()
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    reinforce_loss2 = ((cost2 - bl_val) * log_likelihood2).mean()
    loss = reinforce_loss + reinforce_loss2 + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    # if step % int(opts.log_step) == 0:
    #     log_values(cost, grad_norms, epoch, batch_id, step,
    #                log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)

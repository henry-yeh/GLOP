import sys
sys.path.insert(0, './')
import torch
import time
import argparse
from tqdm import tqdm
from heatmap.cvrp.inst import gen_inst, gen_pyg_data, trans_tsp
from heatmap.cvrp.eval import eval
from heatmap.cvrp.sampler import Sampler
from nets.partition_net import Net
from utils import load_model

EPS = 1e-10
DEVICE = 'cuda:0'
LR = 3e-4
K_SPARSE = {
    1000: 100,
    2000: 200,
}

def infer_heatmap(model, pyg_data):
    heatmap = model(pyg_data)
    heatmap = heatmap / (heatmap.min()+1e-5)
    heatmap = model.reshape(pyg_data, heatmap) + 1e-5
    return heatmap
    
def train_batch(model, optimizer, n, bs, opts):
    model.train()
    loss_lst = []
    for _ in range(opts.batch_size):
        coors, demand, capacity = gen_inst(n, DEVICE)
        pyg_data = gen_pyg_data(coors, demand, capacity, K_SPARSE[n])
        heatmap = infer_heatmap(model, pyg_data)
        sampler = Sampler(demand, heatmap, capacity, bs, DEVICE)
        routes, log_probs = sampler.gen_subsets(require_prob=True)
        tsp_insts, n_tsps_per_route = trans_tsp(coors, routes)
        objs = eval(tsp_insts, n_tsps_per_route, opts)
        baseline = objs.mean()
        log_probs = log_probs.to(DEVICE)
        reinforce_loss = torch.sum((objs-baseline) * log_probs.sum(dim=1)) / bs
        loss_lst.append(reinforce_loss)
    
    loss = sum(loss_lst) / opts.batch_size
    optimizer.zero_grad()
    loss.backward()
    if not opts.no_clip:
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opts.max_norm, norm_type=2)
    optimizer.step()
    
def infer_instance(model, inst, opts):
    model.eval()
    coors, demand, capacity = inst
    n = demand.size(0)-1
    pyg_data = gen_pyg_data(coors, demand, capacity, K_SPARSE[n])
    heatmap = infer_heatmap(model, pyg_data)
    sampler = Sampler(demand, heatmap, capacity, 1, DEVICE)
    routes = sampler.gen_subsets(require_prob=False, greedy_mode=True)
    tsp_insts, n_tsps_per_route = trans_tsp(coors, routes)
    obj = eval(tsp_insts, n_tsps_per_route, opts).min()
    return obj

def train_epoch(n, bs, steps_per_epoch, net, optimizer, scheduler, opts):
    for _ in tqdm(range(steps_per_epoch)):
        train_batch(net, optimizer, n, bs, opts)
    scheduler.step()

@torch.no_grad()
def validation(n, net, opts):
    sum_obj = 0
    for _ in range(opts.val_size):
        inst = gen_inst(n, DEVICE)
        obj = infer_instance(net, inst, opts)
        sum_obj += obj
    avg_obj = sum_obj / opts.val_size
    return avg_obj

def train(n, bs, steps_per_epoch, n_epochs, opts):
    revisers = []
    for reviser_size in opts.revision_lens:
        reviser_path = f'pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt'
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)
    for reviser in revisers:
        reviser.to(DEVICE)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)    
    opts.revisers = revisers
    
    net = Net(opts.units, 3, K_SPARSE[n], 2, depth=opts.depth).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs)
    
    if opts.checkpoint_path == '':
        starting_epoch = 1
    else:
        checkpoint = torch.load(opts.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        
    sum_time = 0
    best_avg_obj = validation(n, net, opts)
    print('epoch 0', best_avg_obj.item())
    for epoch in range(starting_epoch, n_epochs + 1):
        start = time.time()
        train_epoch(n, bs, steps_per_epoch, net, optimizer, scheduler, opts)
        sum_time += time.time() - start
        avg_obj = validation(n, net, opts)
        print(f'epoch {epoch}: ', avg_obj.item())
        if best_avg_obj > avg_obj:
            best_avg_obj = avg_obj
            print(f'Save checkpoint-{epoch}.')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, f'./pretrained/Partitioner/cvrp/cvrp-{n}-{epoch}-cos.pt')
    print('total training duration:', sum_time)
    
if __name__ == '__main__':
    import pprint as pp
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_size', type=int, default=400)
    parser.add_argument('--revision_lens', nargs='+', default=[20] ,type=int,
                        help='The sizes of revisers')
    parser.add_argument('--revision_iters', nargs='+', default=[5], type=int,
                        help='Revision iterations (I_n)')
    parser.add_argument('--decode_strategy', type=str, default='greedy', help='decode strategy of the model')
    parser.add_argument('--width', type=int, default=10)
    parser.add_argument('--no_aug', action='store_true', help='Disable instance augmentation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--steps_per_epoch', type=int, default=256)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--max_norm', type=float, default=1)
    parser.add_argument('--units', type=int, default=48)
    parser.add_argument('--no_clip', action='store_true')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--depth', type=int, default=12)
    opts = parser.parse_args()
    opts.no_aug = True
    opts.no_prune = False
    opts.problem_type = 'tsp'
    
    torch.manual_seed(opts.seed)
    pp.pprint(vars(opts))
    train(opts.problem_size, opts.width, opts.steps_per_epoch, opts.n_epochs, opts)
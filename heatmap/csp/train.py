import sys
sys.path.insert(0, './')
import torch
import time
import argparse
from inst import gen_inst, gen_pyg_data
from eval import eval
from nets.partition_net import Net
from sampler import Sampler
from utils import load_model
import tqdm
import numba

LR = 3e-4
N_VAL = 30
CUDA = "cuda:0"
K_SPARSE = 50


def train_instance(model, optimizer, opts):
    n = opts.problem_size
    coor, dist_mat, cover_map = gen_inst(n)
    pyg_data = gen_pyg_data(coor, dist_mat, cover_map)
    heatmap = model(pyg_data.to(CUDA))
    heatmap = model.reshape(pyg_data, heatmap)
    sampler = Sampler(heatmap.cpu(), cover_map, opts.bs, 'cpu')
    subsets, log_probs = sampler.gen_subsets(require_prob=True)
    objs = eval(subsets.to(CUDA), coor.to(CUDA), opts)
    baseline = objs.mean()
    reinforce_loss = torch.sum((objs-baseline) * log_probs.to(CUDA).sum(dim=1)) / opts.bs
    return reinforce_loss
    
def infer_instance(model, inst, opts):
    model.eval()
    coor, dist_mat, cover_map, pyg_data = inst
    heatmap = model(pyg_data.to(CUDA))
    heatmap = model.reshape(pyg_data, heatmap)
    sampler = Sampler(heatmap.cpu(), cover_map, opts.bs, 'cpu')
    subsets = sampler.gen_subsets(require_prob=False, greedy_mode=False)
    obj = eval(subsets.to(CUDA), coor.to(CUDA), opts).mean()
    return obj

def train_epoch(steps_per_epoch, net, optimizer, scheduler, opts):
    net.train()
    for _ in tqdm.tqdm(range(steps_per_epoch), disable=True):
        batch_loss = []
        for i in range(opts.batch_size):
            loss = train_instance(net, optimizer, opts)
            batch_loss.append(loss)
        optimizer.zero_grad()
        (sum(batch_loss) / opts.batch_size).backward() 
        optimizer.step()
    scheduler.step()
        
@torch.no_grad()
def validation(net, inst_list, opts):
    sum_obj = 0
    for inst in inst_list:
        obj = infer_instance(net, inst, opts)
        sum_obj += obj
    avg_obj = sum_obj / N_VAL
    return avg_obj

def train(steps_per_epoch, n_epochs, opts):
    n = opts.problem_size
    
    revisers = []
    for reviser_size in opts.revision_lens:
        reviser_path = f'pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt'
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)
    for reviser in revisers:
        reviser.to(CUDA)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)    
    opts.revisers = revisers
    
    net = Net(32, 1, K_SPARSE, edge_feats=2, depth=12).to(CUDA)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs, eta_min=1e-6)
    
    val_list = [gen_inst(n) for _ in range(N_VAL)]
    for idx, inst in enumerate(val_list):
        coor, dist_mat, cover_map = inst
        pyg_data = gen_pyg_data(coor, dist_mat, cover_map)
        val_list[idx] = (coor, dist_mat, cover_map, pyg_data)
        
    best_avg_obj = validation(net, val_list, opts)
    print('epoch 0: ', best_avg_obj.item())
    sum_time = 0
    for epoch in range(1, n_epochs+1):
        start = time.time()
        train_epoch(steps_per_epoch, net, optimizer, scheduler, opts)
        sum_time += time.time() - start
        avg_obj = validation(net, val_list, opts)
        print(f'epoch {epoch}: ', avg_obj.item())
        if best_avg_obj > avg_obj:
            best_avg_obj = avg_obj
            torch.save(net.state_dict(), f'./pretrained/Partitioner/csp/csp-{n}-{epoch}.pt')
    print('total training duration:', sum_time)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_size', type=int, default=300)
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--revision_lens', nargs='+', default=[20] ,type=int,
                        help='The sizes of revisers')
    parser.add_argument('--revision_iters', nargs='+', default=[5], type=int,
                        help='Revision iterations (I_n)')
    parser.add_argument('--decode_strategy', type=str, default='greedy', help='decode strategy of the model')
    # parser.add_argument('--width', type=int, default=50, 
    #                     help='The initial solutions for a TSP instance generated with diversified insertion')
    parser.add_argument('--no_aug', action='store_true', help='Disable instance augmentation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--bs', type=int, default=20, help="Sampling width for Training")
    parser.add_argument('--batch_size', type=int, default=5, help="Training batch size")
    
    opts = parser.parse_args()
    opts.no_aug = True
    opts.no_prune = False
    opts.problem_type = 'tsp'
    opts.eval_batch_size = opts.bs
    
    torch.manual_seed(opts.seed)

    train(256, 200, opts) 

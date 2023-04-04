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

EPS = 1e-10
DEVICE = 'cuda:1'
LR = 1e-4
N_VAL = 100

k_sparse_table = {
    20: 5,
    50: 20,
    100: 20,
    500: 50,
    1000: 100,
    5000: 200,
    10000: 500
}

def train_instance(model, optimizer, n, bs, opts):
    model.train()
    coor, dist_mat, prizes, penalties = gen_inst(n, DEVICE)
    pyg_data = gen_pyg_data(prizes, penalties, dist_mat, k_sparse_table[n])
    heatmap = model.reshape(pyg_data, model(pyg_data)) + EPS
    sampler = Sampler(prizes, heatmap, bs, DEVICE)
    subsets, log_probs = sampler.gen_subsets(require_prob=True)
    sol_penalty = sampler.gen_penalty(subsets, penalties)
    opts.eval_batch_size = bs
    objs = eval(subsets, coor, sol_penalty, opts)
    baseline = objs.mean()
    reinforce_loss = torch.sum((objs-baseline) * log_probs.sum(dim=1)) / bs
    optimizer.zero_grad()
    reinforce_loss.backward()
    optimizer.step()
    
def infer_instance(model, inst, width, opts):
    model.eval()
    coor, dist_mat, prizes, penalties = inst
    n = prizes.size(0)-1
    pyg_data = gen_pyg_data(prizes, penalties, dist_mat, k_sparse_table[n])
    heatmap = model.reshape(pyg_data, model(pyg_data)) + EPS
    sampler = Sampler(prizes, heatmap, width, DEVICE)
    subsets = sampler.gen_subsets(require_prob=False)
    sol_penalty = sampler.gen_penalty(subsets, penalties)
    opts.eval_batch_size = width
    obj = eval(subsets, coor, sol_penalty, opts).min()
    return obj

def train_epoch(n, bs, steps_per_epoch, net, optimizer, scheduler, opts):
    for _ in range(steps_per_epoch):
        train_instance(net, optimizer, n, bs, opts)
        scheduler.step()
        
@torch.no_grad()
def validation(n, width, net, opts):
    sum_obj = 0
    for _ in range(N_VAL):
        inst = gen_inst(n, DEVICE)
        obj = infer_instance(net, inst, width, opts)
        sum_obj += obj
    avg_obj = sum_obj / N_VAL
    return avg_obj

def train(n, bs, val_width, steps_per_epoch, n_epochs, opts):
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
    
    net = Net(32, 2, k_sparse_table[n]).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs)
    
    best_avg_obj = validation(n, val_width, net, opts)
    print('epoch 0: ', best_avg_obj.item())
    sum_time = 0
    for epoch in range(1, n_epochs+1):
        start = time.time()
        train_epoch(n, bs, steps_per_epoch, net, optimizer, scheduler, opts)
        sum_time += time.time() - start
        avg_obj = validation(n, val_width, net, opts)
        print(f'epoch {epoch}: ', avg_obj.item())
        if best_avg_obj > avg_obj:
            best_avg_obj = avg_obj
            torch.save(net.state_dict(), f'./pretrained/Partitioner/pctsp/pctsp-{n}-{epoch}.pt')
    print('total training duration:', sum_time)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_size', type=int, default=500)
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--revision_lens', nargs='+', default=[] ,type=int,
                        help='The sizes of revisers')
    parser.add_argument('--revision_iters', nargs='+', default=[], type=int,
                        help='Revision iterations (I_n)')
    parser.add_argument('--decode_strategy', type=str, default='greedy', help='decode strategy of the model')
    parser.add_argument('--width', type=int, default=50, 
                        help='The initial solutions for a TSP instance generated with diversified insertion')
    parser.add_argument('--no_aug', action='store_true', help='Disable instance augmentation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    opts = parser.parse_args()
    opts.no_aug = True
    opts.problem_type = 'tsp'
    
    torch.manual_seed(opts.seed)

    train(opts.problem_size, opts.width, 5, 512, 10, opts) 
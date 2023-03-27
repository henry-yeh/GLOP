import sys
sys.path.insert(0, './')
import torch
import time

from inst import gen_inst, gen_pyg_data
from nets.partition_net import Net
from sampler import Sampler

EPS = 1e-10
DEVICE = 'cuda:0'
BS = 20
LR = 1e-4
k_sparse_table = {
    20: 5,
    500: 50
}


def toy_obj(subsets, coor, sol_penalty):
    return torch.count_nonzero(subsets, dim=1).float()

def train_instance(model, optimizer, n, bs):
    model.train()
    coor, dist_mat, prizes, penalties = gen_inst(n, DEVICE)
    pyg_data = gen_pyg_data(prizes, penalties, dist_mat, k_sparse_table[n])
    heatmap = model.reshape(pyg_data, model(pyg_data)) + EPS
    sampler = Sampler(prizes, heatmap, bs, DEVICE)
    subsets, log_probs = sampler.gen_subsets(require_prob=True)
    sol_penalty = sampler.gen_penalty(subsets, penalties)
    objs = toy_obj(subsets, coor, sol_penalty)
    baseline = objs.mean()
    reinforce_loss = torch.sum((objs-baseline) * log_probs.sum(dim=1)) / bs
    optimizer.zero_grad()
    reinforce_loss.backward()
    optimizer.step()
    
def infer_instance(model, inst, width):
    model.eval()
    coor, dist_mat, prizes, penalties = inst
    n = prizes.size(0)-1
    pyg_data = gen_pyg_data(prizes, penalties, dist_mat, k_sparse_table[n])
    heatmap = model.reshape(pyg_data, model(pyg_data)) + EPS
    sampler = Sampler(prizes, heatmap, width, DEVICE)
    subsets = sampler.gen_subsets(require_prob=False)
    sol_penalty = sampler.gen_penalty(subsets, penalties)
    obj = toy_obj(subsets, coor, sol_penalty).min()
    return obj

def train_epoch(n, bs, steps_per_epoch, net, optimizer):
    for _ in range(steps_per_epoch):
        train_instance(net, optimizer, n, bs)
        
@torch.no_grad()
def validation(n, width, net):
    sum_obj = 0
    n_inst = 100
    for _ in range(n_inst):
        inst = gen_inst(n, DEVICE)
        obj = infer_instance(net, inst, width)
        sum_obj += obj
    avg_obj = sum_obj / n_inst
    return avg_obj

def train(n, bs, val_width, steps_per_epoch, n_epochs):
    net = Net(2, k_sparse_table[n]).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    avg_obj = validation(n, val_width, net)
    print('epoch 0: ', avg_obj)
    sum_time = 0
    for epoch in range(1, n_epochs+1):
        start = time.time()
        train_epoch(n, bs, steps_per_epoch, net, optimizer)
        sum_time += time.time() - start
        avg_obj = validation(n, val_width, net)
        print(f'epoch {epoch}: ', avg_obj)
        torch.save(net.state_dict(), f'./pretrained/Partitioner/pctsp/pctsp-{n}-{epoch}.pt')
    
    
if __name__ == '__main__':
    train(500, 50, 20, 128, 5)
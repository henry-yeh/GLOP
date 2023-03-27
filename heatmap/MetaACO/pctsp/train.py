import sys
sys.path.insert(0, '../../')

import torch
import time

from inst import gen_inst, gen_pyg_data, load_val_dataset
from nets.partition_net import Net
from sampler import Sampler

EPS = 1e-10
DEVICE = 'cuda:0'
BS = 20
LR = 3e-4
k_sparse_table = {
    20: 5
}


def toy_obj(subsets):
    return torch.count_nonzero(subsets, dim=1)
    

def train_instance(model, optimizer, n, bs):
    model.train()
    dist_mat, prizes, penalties = gen_inst(n, DEVICE)
    pyg_data = gen_pyg_data(prizes, penalties, dist_mat, k_sparse_table[n])
    heatmap = model.reshape(model(pyg_data)) + EPS
    sampler = Sampler(prizes, heatmap, bs, DEVICE)
    subsets, log_probs = sampler.gen_subsets(require_prob=True)
    objs = toy_obj(subsets)
    baseline = objs.mean()
    reinforce_loss = torch.sum((objs-baseline) * log_probs.sum(dim=1)) / bs
    optimizer.zero_grad()
    reinforce_loss.backward()
    optimizer.step()
    
def infer_instance(model, inst, width):
    model.eval()
    dist_mat, prizes, penalties = inst
    n = prizes.size(0)
    pyg_data = gen_pyg_data(prizes, penalties, dist_mat, k_sparse_table[n])
    heatmap = model.reshape(model(pyg_data)) + EPS
    sampler = Sampler(prizes, heatmap, width, DEVICE)
    subsets = sampler.gen_subsets(require_prob=False)
    obj = toy_obj(subsets).min()
    return obj

def train_epoch(n, bs, steps_per_epoch, net, optimizer):
    for _ in range(steps_per_epoch):
        train_instance(net, optimizer, n, bs)
        
@torch.no_grad()
def validation(dataset, width, net):
    sum_obj = 0
    n_inst = len(dataset)
    for inst in dataset:
        obj = infer_instance(net, inst, width)
        sum_obj += obj
    avg_obj = sum_obj / n_inst
    return avg_obj

def train(n, bs, val_width, steps_per_epoch, n_epochs):
    net = Net().to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    val_dataset = load_val_dataset(n, DEVICE)
    best_avg_obj = validation(val_dataset, val_width, net)
    print('epoch 0: ', best_avg_obj)
    sum_time = 0
    for epoch in range(1, n_epochs+1):
        start = time.time()
        train_epoch(n, bs, steps_per_epoch, net, optimizer)
        sum_time += time.time() - start
        avg_obj = validation(val_dataset, val_width, net)
        print(f'epoch {epoch}: ', avg_obj)
        if best_avg_obj > avg_obj:
            best_avg_obj = avg_obj
            torch.save(net.state_dict(), f'./pretrained/Partitioner/pctsp/pctsp-{n}.pt')
    
    

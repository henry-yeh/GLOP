import sys
sys.path.insert(0, './')
import pickle
import torch
import numpy as np
from heatmap.cvrp.infer import infer, load_partitioner
from heatmap.cvrp.sampler import Sampler
from heatmap.cvrp.inst import trans_tsp
  
def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
    
def concat_list(depot_coor, coors, demand, opts):
    coor =  torch.cat([torch.tensor(depot_coor, device=opts.device).unsqueeze(0), 
                    torch.tensor(coors, device=opts.device)], dim=0) # 1+p_size, 2 
    demand = torch.cat([torch.zeros((1), device=opts.device), 
                            torch.tensor(demand, device=opts.device)]) # 1+p_size
    return coor, demand

def add_padding(pi_all, max_seq_len, opts):
    ret = []
    for subset_pi in pi_all:
        assert subset_pi.size(0) == opts.n_subset
        diff = max_seq_len - subset_pi.size(-1)
        subset_pi = torch.cat([subset_pi, torch.zeros((opts.n_subset, diff), dtype=torch.int64 ,device=subset_pi.device)], dim=1)
        ret.append(subset_pi)
    ret = torch.cat(ret, dim=0) # n_val*n_subset, max_len
    assert ret.shape == (opts.val_size * opts.n_subset, max_seq_len)
    return ret
        
def init(path, opts, partitioner=None):
    data = load_dataset(path)
    greedy_mode = True if opts.n_partition == 1 else False
    partitioner = load_partitioner(opts.problem_size, opts.device, opts.ckpt_path) if partitioner is None else partitioner
    dataset = []
    n_tsps_per_route_lst = []
    for inst_id, inst in enumerate(data[:opts.val_size]):
        depot_coor, coors, demand, capacity = inst
        coors, demand = concat_list(depot_coor, coors, demand, opts)
        k_sparse = None if partitioner is None else 300
        heatmap = infer(partitioner, coors, demand, capacity, k_sparse)
        sampler = Sampler(demand, heatmap, capacity, opts.n_partition, 'cpu')
        routes = sampler.gen_subsets(require_prob=False, greedy_mode=greedy_mode) # n_partition, max_len
        assert routes.size(0) == opts.n_partition
        tsp_insts, n_tsps_per_route = trans_tsp(coors.cpu(), routes)
        assert tsp_insts.size(0) == sum(n_tsps_per_route)
        dataset.append(tsp_insts)
        n_tsps_per_route_lst.append(n_tsps_per_route)
    return dataset, n_tsps_per_route_lst
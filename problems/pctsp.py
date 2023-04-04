import sys
sys.path.insert(0, './')
import pickle
import torch
from heatmap.pctsp.infer import infer, load_partitioner
from heatmap.pctsp.sampler import Sampler
  
def load_dataset(path='./data/pctsp/pctsp500_test_seed1234.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
    
def concat_list(depot_coor, coors, penalty, prize, opts):
    coor =  torch.cat([torch.tensor(depot_coor, device=opts.device).unsqueeze(0), 
                    torch.tensor(coors, device=opts.device)], dim=0) # 1+p_size, 2 
    penalty = torch.cat([torch.zeros((1), device=opts.device), 
                            torch.tensor(penalty, device=opts.device)]) # 1+p_size
    prize = torch.cat([torch.zeros((1), device=opts.device), 
                            torch.tensor(prize, device=opts.device)]) # 1+p_size
    return coor, penalty, prize

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
        
def init(path, opts):
    data = load_dataset(path)
    partitioner = load_partitioner(opts.problem_size, opts.device)
    dataset = []
    pi_all = []
    penalty_all = []
    max_seq_len = 0
    for inst_id, inst in enumerate(data[:opts.val_size]):
        depot_coor, coors, penalty, prize, _ = inst
        coors, penalty, prize = concat_list(depot_coor, coors, penalty, prize, opts)
        dataset.append(coors)
        heatmap = infer(partitioner, prize, penalty, coors)
        sampler = Sampler(prize, heatmap, opts.n_subset, opts.device)
        subset = sampler.gen_subsets(require_prob=False, greedy_mode=False) # n_subset, max_len
        assert subset.size(0) == opts.n_subset
        if subset.size(1) > max_seq_len:
            max_seq_len = subset.size(1)
        penalty = sampler.gen_penalty(subset, penalty) # n_subset
        pi_all.append(subset)  # pi_all: list, (val_size, n_subset, max_len)
        penalty_all.append(penalty) # list, (val_size, n_subset)
    pi_all = add_padding(pi_all, max_seq_len, opts) # n_val*n_subset, max_len
    
    # transform into TSPs
    dataset = torch.stack(dataset) # val_size, p+1, 2
    dataset = torch.repeat_interleave(dataset, opts.n_subset, 0) # n_val*n_subset, p+1, 2
    seed = dataset.gather(1, pi_all.unsqueeze(-1).repeat(1, 1, 2)) # (n_val*n_subset, max_seq_len, 2)
    penalty = torch.cat(penalty_all, dim=0)
    return seed, penalty # (n_val*n_subset, max_seq_len, 2), (val_size*n_subset, )

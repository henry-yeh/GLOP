import torch
from torch.distributions import Categorical
import numpy as np
from copy import deepcopy

class Sampler():
    '''
    To sample node subsets given a PCTSP heatmap.
    '''
    def __init__(self, prizes, heatmap, bs=20, device='cpu'):
        self.n = prizes.size(0)
        self.prizes = prizes
        self.heatmap = heatmap
        self.min_prizes = self.n / 4
        self.bs = bs
        self.ants_idx = torch.arange(bs)
        self.device = device
    
    def gen_subsets(self, require_prob=False):
        solutions = []
        log_probs_list = []
        cur_node = torch.zeros(size=(self.bs,), dtype=torch.int64, device=self.device)
        visit_mask = torch.ones(size=(self.bs, self.n), device=self.device) # 1) mask the visted regular node; 2) once return to depot, mask all
        depot_mask = torch.ones(size=(self.bs, self.n), device=self.device) 
        depot_mask[: , 0] = 0 # unmask the depot when 1) enough prize collected; 2) all nodes visited
        
        collected_prize = torch.zeros(size=(self.bs,), device=self.device)
        done = False
        # construction
        while not done:
            cur_node, log_prob = self.pick_node(visit_mask, depot_mask, cur_node, require_prob) # pick action
            # update solution and log_probs
            solutions.append(cur_node) 
            log_probs_list.append(log_prob)
            # update collected_prize and mask
            collected_prize += self.prizes[cur_node]
            if require_prob:
                visit_mask = visit_mask.clone()
                depot_mask = depot_mask.clone()
            visit_mask, depot_mask = self.update_mask(visit_mask, depot_mask, cur_node, collected_prize)
            # check done
            done = self.check_done(cur_node)
        if require_prob:
            return torch.stack(solutions).permute(1, 0), torch.stack(log_probs_list).permute(1, 0)  # shape: [bs, max_seq_len]
        else:
            return torch.stack(solutions).permute(1, 0)
    
    def pick_node(self, visit_mask, depot_mask, cur_node, require_prob):
        heatmap = self.heatmap[cur_node] 
        dist = (heatmap * visit_mask * depot_mask)
        dist = Categorical(dist)
        item = dist.sample()
        log_prob = dist.log_prob(item) if require_prob else None
        return item, log_prob  # (bs,)
    
    def update_mask(self, visit_mask, depot_mask, cur_node, collected_prize):
        # mask regular visted node
        visit_mask[self.ants_idx, cur_node] = 0
        # if at depot, mask all regular nodes, and unmask depot
        at_depot = cur_node == 0
        visit_mask[at_depot, 0] = 1
        visit_mask[at_depot, 1:] = 0
        # unmask the depot for in either case
        # 1) not at depot and enough prize collected
        depot_mask[(~at_depot) * (collected_prize > self.min_prizes), 0] = 1
        # 2) not at depot and all nodes visited
        depot_mask[(~at_depot) * ((visit_mask[:, 1:]==0).all(dim=1)), 0] = 1
        return visit_mask, depot_mask

    def check_done(self, cur_node):
        # is all at depot ?
        return (cur_node == 0).all()        

if __name__ == '__main__':
    torch.set_printoptions(precision=4,sci_mode=False)
    sampler = Sampler(prizes=torch.rand((10, )), heatmap=torch.ones((10, 10)))
    sols, probs = sampler.gen_subsets(require_prob=1)
    print(sols)
    
    
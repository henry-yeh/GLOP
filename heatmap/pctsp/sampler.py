import torch
from torch.distributions import Categorical

class Sampler():
    '''
    To sample node subsets given a PCTSP heatmap.
    '''
    def __init__(self, prizes, heatmap, bs=20, device='cpu'):
        self.n = prizes.size(0)
        self.prizes = prizes
        self.heatmap = heatmap
        self.min_prizes = 1
        self.bs = bs
        self.ants_idx = torch.arange(bs)
        self.device = device
    
    def gen_subsets(self, require_prob=False, greedy_mode=False):
        if greedy_mode:
            assert not require_prob
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
            cur_node, log_prob = self.pick_node(visit_mask, depot_mask, cur_node, require_prob, greedy_mode) # pick action
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

    def gen_penalty_bool(self, sol, n):
        '''
        Args:
            sol: (width, max_seq_len)
        '''
        width = sol.size(0)
        seq_len = sol.size(1)
        expanded_nodes = torch.arange(n, device=self.device).repeat(width, seq_len, 1) # (width, seq_len, n)
        expanded_sol = torch.repeat_interleave(sol, n, dim=-1).reshape(width, seq_len, n)
        return (torch.eq(expanded_nodes, expanded_sol)==0).all(dim=1)
    
    def gen_penalty(self, solutions, node_penalty):
        '''
        Args:
            solutions: (width, max_len)
        '''
        penalty_bool = self.gen_penalty_bool(solutions, self.n)
        sols_penalty = []
        for idx in range(solutions.size(0)):
            penalty = node_penalty[penalty_bool[idx]].sum()
            sols_penalty.append(penalty)
        return torch.stack(sols_penalty)
    
    def pick_node(self, visit_mask, depot_mask, cur_node, require_prob, greedy_mode=False):
        log_prob = None
        heatmap = self.heatmap[cur_node] 
        dist = (heatmap * visit_mask * depot_mask)
        if not greedy_mode:
            dist = Categorical(dist)
            item = dist.sample()
            log_prob = dist.log_prob(item) if require_prob else None
        else:
            item, _ = dist.max(dim=1)
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
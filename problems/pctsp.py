import pickle
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.distributions import Categorical

class Constructor():
    def __init__(self, coors, penalty, prize, opts, test_mood=False):
        self.opts = opts
        self.n = len(penalty)
        self.distances = self.gen_dist(coors)
        self.penalty = penalty
        self.prize = prize
        self.gen_heu()
        self.device = opts.device
        self.min_prizes = 1
        self.width = opts.n_subset
        self.test_mood = test_mood
        self.arange = torch.arange(self.width)
        
    def gen_heu(self, temperature=1):
        if self.opts.problem_type == 'pctsp':
            exp = self.prize.repeat(self.n, 1) * self.penalty.repeat(self.n, 1)
        elif self.opts.problem_type == 'spctsp':
            exp = self.penalty.repeat(self.n, 1)
        else:
            raise NotImplementedError
        exp[torch.arange(self.n), torch.arange(self.n)] = 0
        exp[:, 0] = 0
        baseline = exp.sum() / ((self.n-1)**2 + 1)
        exp[torch.arange(self.n), torch.arange(self.n)] = baseline
        exp[:, 0] = baseline
        exp -= baseline
        self.heuristic = torch.softmax(exp/temperature, dim=1,)
    
    def gen_dist(self, coordinates):
        distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
        return distances # diagonal is zero
    
    @torch.no_grad()
    def construction(self):
        # order-agnostic, just pick a subset of nodes
        cur_node = torch.zeros(size=(self.width,), dtype=torch.int64, device=self.device)
        solutions = [cur_node]
        
        visit_mask = torch.ones(size=(self.width, self.n), device=self.device) # 1) mask the visted regular node; 2) once return to depot, mask all
        depot_mask = torch.ones(size=(self.width, self.n), device=self.device) 
        depot_mask[: , 0] = 0 # unmask the depot when 1) enough prize collected; 2) all nodes visited
        
        collected_prize = torch.zeros(size=(self.width,), device=self.device)
        # construction
        for _ in range(self.n-1):
            cur_node= self.pick_node(visit_mask, depot_mask, cur_node) # pick action
            # update solution and log_probs
            solutions.append(cur_node) 
            # update collected_prize and mask
            collected_prize += self.prize[cur_node]
            visit_mask, depot_mask = self.update_mask(visit_mask, depot_mask, cur_node, collected_prize)
            # check done
        sols = torch.stack(solutions).permute(1, 0)
        penalty = self.gen_penalty(sols)
        return sols, penalty
        
    
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
    
    def gen_penalty(self, solutions):
        '''
        Args:
            solutions: (width, max_len)
        '''
        if self.test_mood:
            u = solutions
            v = torch.roll(u, shifts=-1, dims=1)
            length = torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)
        penalty_bool = self.gen_penalty_bool(solutions, self.n)
        penalty = []
        for ant_id in range(self.width):
            ant_penalty = self.penalty[penalty_bool[ant_id]].sum()
            penalty.append(ant_penalty)
        if self.test_mood:
            return length, torch.stack(penalty)
        else:
            return torch.stack(penalty)
    
    def pick_node(self, visit_mask, depot_mask, cur_node):
        heuristic = self.heuristic[cur_node] 
        dist = heuristic * visit_mask * depot_mask
        dist = Categorical(dist)
        item = dist.sample()
        return item
    
    def update_mask(self, visit_mask, depot_mask, cur_node, collected_prize):
        # mask regular visted node
        visit_mask[self.arange, cur_node] = 0
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

def disturb_dummy_coor(pi_batch, seeds):
    '''
    Args:
        pi_batch: (x, p_size+1)
        seeds: (x, p_size+1, 2)
    '''
    zeroIndices = torch.nonzero(pi_batch.eq(0))
    seeds[zeroIndices[:, 0], zeroIndices[:, 1]] += 1e-5*((torch.rand((zeroIndices.size(0), 2), device=seeds.device))+0.1)
    return seeds

def init(path, opts):
    data = load_dataset(path)
    dataset = []
    pi_all = []
    penalty_all = []
    test_objs = []
    dataloader = DataLoader(data, batch_size=1)
    for inst_id, inst in enumerate(data[:opts.val_size]):
        depot_coor, coors, penalty, prize, _ = inst
        coors, penalty, prize = concat_list(depot_coor, coors, penalty, prize, opts)
        dataset.append(coors)
        cons = Constructor(coors, penalty, prize, opts)
        sols, penalty = cons.construction()
        pi_all.append(sols)
        penalty_all.append(penalty)
    
    # transform into TSPs
    dataset = torch.stack(dataset) # val_size, p_size, 2
    dataset = dataset.repeat(opts.n_subset, 1, 1) # width*val_size, p_size, 2
    pi_all = torch.stack(pi_all).permute(1,0,2).contiguous().reshape(opts.n_subset*opts.val_size, opts.problem_size+1) # width*val_size, p_size+1
    seed = dataset.gather(1, pi_all.unsqueeze(-1).expand_as(dataset)) # (width*val_size, p_size+1, 2). The first is depot
    penalty = torch.stack(penalty_all).permute(1,0) # (width, val_size)
    seed = disturb_dummy_coor(pi_all, seed)
    return seed, penalty
        
if __name__ == '__main__':
    import argparse
    torch.set_printoptions(12)
    data = load_dataset()
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", type=int, default=500)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--val_size", type=int, default=2)
    opts = parser.parse_args()
    opts.device = 'cpu'
    seed, penalty = init('./data/pctsp/pctsp500_test_seed1234.pkl', 10, opts)
    print(seed[0, -50:], penalty.shape)
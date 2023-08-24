import torch
from torch.distributions import Categorical

class Sampler():

    def __init__(self, heatmap, cover_map, bs=20, device='cpu'):
        self.n = heatmap.size(0) # including dummy node (depot)
        self.heatmap = heatmap
        self.cover_map = cover_map # tensor, (n, #covered_nodes), cover_map[0] = [0, 0, .., 0]
        self.k = cover_map.size(1)
        self.bs = bs
        self.ants_idx = torch.arange(bs)
        self.device = device
    
    def gen_subsets(self, require_prob=False, greedy_mode=False):
        if greedy_mode:
            assert not require_prob
        solutions = []
        log_probs_list = []
        cur_node = torch.zeros(size=(self.bs,), dtype=torch.int64, device=self.device) # dummy node indexed by 0
        visit_mask = torch.ones(size=(self.bs, self.n), dtype=torch.bool, device=self.device)
        visit_mask[:, 0] = 0 # don't allow returning to depot untill all nodes covered
        
        is_covered = torch.zeros(size=(self.bs, self.n), dtype=torch.bool, device=self.device) # binary indicators of whether a city has been covered
        is_covered[:, 0] = 1 # the dummy depot has been covered
        done = False
        # construction
        while not done:
            cur_node, log_prob = self.pick_node(visit_mask, cur_node, require_prob, greedy_mode) # pick action
            # update solution and log_probs
            solutions.append(cur_node) 
            log_probs_list.append(log_prob)
            
            if require_prob:
                visit_mask = visit_mask.clone()
            
            is_covered = self.update_cover(cur_node, is_covered, self.cover_map)
            visit_mask = self.update_mask(visit_mask, cur_node, is_covered)
            
            # check done
            done = self.check_done(cur_node)
            
        if require_prob:
            return torch.stack(solutions).permute(1, 0), torch.stack(log_probs_list).permute(1, 0)  # shape: [bs, max_seq_len]
        else:
            return torch.stack(solutions).permute(1, 0)

    def pick_node(self, visit_mask, cur_node, require_prob, greedy_mode=False):
        log_prob = None
        heatmap = self.heatmap[cur_node] 
        dist = heatmap * visit_mask
        if not greedy_mode:
            dist = Categorical(dist)
            item = dist.sample()
            log_prob = dist.log_prob(item) if require_prob else None
        else:
            _, item = dist.max(dim=1)
        return item, log_prob  # (bs,)
    
    def update_cover(self, cur_node: torch.Tensor, is_covered: torch.Tensor, cover_map: torch.Tensor):
        # cover cur_node
        is_covered[self.ants_idx, cur_node] = 1
        # cover knn, cover map: (n, k)
        knn = cover_map[cur_node] # (bs, k)
        is_covered[torch.repeat_interleave(self.ants_idx, self.k), knn.view(-1)] = 1
        return is_covered
    
    def update_mask(self, visit_mask, cur_node, is_covered):
        # mask regular visted node
        visit_mask[self.ants_idx, cur_node] = 0
        # if at depot, mask all regular nodes, and unmask depot
        at_depot = cur_node == 0
        visit_mask[at_depot, 0] = 1
        visit_mask[at_depot, 1:] = 0
        # if not at depot and all nodes covered:
        back = (~at_depot) * ((is_covered==1).all(dim=1))
        # (1) unmask the depot 
        visit_mask[back, 0] = 1
        # (2) mask all other nodes
        visit_mask[back, 1:] = 0
        return visit_mask

    def check_done(self, cur_node):
        # all at depot ?
        return (cur_node == 0).all()
    

if __name__ == '__main__':
    n = 100
    k = 7
    coor = torch.rand((n+1, 2))
    coor[0] += 2
    distances = torch.norm(coor[:, None] - coor, dim=2, p=2)
    distances[torch.arange(n+1), torch.arange(n+1)] = 1e5 # note here
    topk_values, topk_indices = torch.topk(distances, 
                                            k=k, 
                                            dim=1, largest=False)
    sampler = Sampler(1/distances, cover_map=topk_indices, bs=20)
    sols, log_probs = sampler.gen_subsets(require_prob=True)
    print("sols: ", sols)
    print("log probs: ", log_probs)
    
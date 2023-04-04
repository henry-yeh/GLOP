import torch
from torch.distributions import Categorical
import math

class Sampler():
    def __init__(self, demand, heatmap, capacity, bs, device):
        self.n = demand.size(0)
        self.demand = demand.to(device)
        self.heatmap = heatmap.to(device)
        self.capacity = capacity
        self.max_vehicle = math.ceil(sum(self.demand)/capacity) + 1 
        self.total_demand = self.demand.sum()
        self.bs = bs
        self.ants_idx = torch.arange(bs)
        self.device = device
    
    def gen_subsets(self, require_prob=False, greedy_mode=False):
        if greedy_mode:
            assert not require_prob
        actions = torch.zeros((self.bs,), dtype=torch.long ,device=self.device)
        visit_mask = torch.ones(size=(self.bs, self.n), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.bs,), device=self.device)
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
        
        vehicle_count = torch.zeros((self.bs,), device=self.device)
        demand_count = torch.zeros((self.bs,), device=self.device)
        depot_mask, vehicle_count, demand_count = self.update_depot_mask(vehicle_count, demand_count, actions, capacity_mask, visit_mask)
        
        paths_list = [actions]
        log_probs_list = []
        done = self.check_done(visit_mask, actions)
        while not done:
            actions, log_probs = self.pick_node(actions, visit_mask, capacity_mask, depot_mask, require_prob, greedy_mode)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
                depot_mask = depot_mask.clone()
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            depot_mask, vehicle_count, demand_count = self.update_depot_mask(vehicle_count, demand_count, actions, capacity_mask, visit_mask) 
            done = self.check_done(visit_mask, actions)
        if require_prob:
            return torch.stack(paths_list).permute(1, 0), torch.stack(log_probs_list).permute(1, 0)
        else:
            return torch.stack(paths_list).permute(1, 0)
    
    def pick_node(self, prev, visit_mask, capacity_mask, depot_mask, require_prob, greedy_mode=False):
        log_prob = None
        heatmap = self.heatmap[prev] 
        dist = (heatmap * visit_mask * capacity_mask * depot_mask)
        if not greedy_mode:
            try:
                dist = Categorical(dist)
                item = dist.sample()
                log_prob = dist.log_prob(item) if require_prob else None
            except:
                dist = torch.softmax(torch.log(dist), dim=1)
                item = torch.multinomial(dist, num_samples=1).squeeze()
                log_prob= torch.log(dist[torch.arange(self.bs),item])
        else:
            _, item = dist.max(dim=1)
        return item, log_prob
    
    def update_depot_mask(self, vehicle_count, demand_count, actions, capacity_mask, visit_mask):
        depot_mask = torch.ones((self.bs, self.n), device=self.device)
        # update record
        vehicle_count[actions==0] += 1
        demand_count += self.demand[actions]
        remaining_demand = self.total_demand - demand_count
        # mask
        depot_mask[remaining_demand > self.capacity*(self.max_vehicle-vehicle_count), 0] = 0
        # unmask
        depot_mask[((visit_mask[:, 1:]*capacity_mask[:, 1:])==0).all(dim=1), 0] = 1
        # depot_mask[(capacity_mask[:, 1:]==0).all(dim=1), 0] = 1
        return depot_mask, vehicle_count, demand_count
    
    def update_visit_mask(self, visit_mask, actions):
        visit_mask[torch.arange(self.bs, device=self.device), actions] = 0
        visit_mask[:, 0] = 1 # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0 # one exception is here
        return visit_mask
    
    def update_capacity_mask(self, cur_nodes, used_capacity):
        capacity_mask = torch.ones(size=(self.bs, self.n), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (bs,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.n) # (bs, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.bs, 1) # (bs, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0
        
        return used_capacity, capacity_mask
    
    def check_done(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()
    
    
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from utils import load_model
from torch.utils.data import DataLoader
import time
from utils.functions import reconnect
from utils.functions import load_problem
from problems.tsp.tsp_baseline import solve_insertion
import pprint as pp
from utils.functions import decomposition, revision

mp = torch.multiprocessing.get_context('spawn')

class Q_learning:
    def __init__(self, opts, revisers):
        '''
        Action range for Reviser_n should be 1 ~ n//2.
        State range for Reviser_n should be 1 ~ I_n, 
        where I_n is the terminating state with Q-value = 0 for all actions.
        Size of Q_n should be (I_n, n//2).
        '''

        pp.pprint(vars(opts))

        self.revisers = revisers
        self.reviser_sizes = opts.revision_lens
        self.revision_iters = opts.revision_iters
        assert len(self.reviser_sizes) == len(self.revision_iters), 'Invalid input !'
        self.M = len(self.reviser_sizes)
        self.width = opts.width
        self.epsilon = opts.epsilon
        self.alpha = opts.alpha
        self.gamma = opts.gamma
        self.batch_size = opts.batch_size
        self.max_iter = opts.max_iter

        self.q_tables = []
        self.initialize_tables()
    
    def initialize_tables(self):
        for i in range(self.M):
            n_states = self.revision_iters[i]
            n_actions = self.reviser_sizes[i] // 2
            Q = torch.zeros(size=(n_states, n_actions))
            self.q_tables.append(Q)
    
    def sampleTSPsize(self):
        # return 200
        return torch.randint(low=200, high=1001, size=(1,)).item()
    
    def sampleTSPInstance(self, n):
        return torch.rand(size=(self.batch_size, n, 2))
    
    def initializeTours(self, batch):
        pi_batch = torch.LongTensor(size=(batch.shape[: 2]))
        avg_cost = 0
        for instance_id, instance in enumerate(batch):

            cost, pi, duration = solve_insertion(
                                    directory=None, 
                                    name=None, 
                                    loc=instance,
                                    method='farthest',
                                    )
            avg_cost += cost
            pi_batch[instance_id] = torch.tensor(pi)
        avg_cost /= batch.shape[0]
        seed = batch.gather(1, pi_batch.unsqueeze(-1).expand_as(batch)).to(device)
        if self.width > 1:
            assert self.width % 4 == 0, 'Width must be an integral multiple of 4 for valid augmentation !'
            aug_shift = self.width // 4
            seed = torch.cat([torch.roll(seed, i, 1) for i in range(0, aug_shift)], dim=0)
            seed2 = torch.cat((1 - seed[:, :, [0]], seed[:, :, [1]]), dim=2)
            seed3 = torch.cat((seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
            seed4 = torch.cat((1 - seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
            seed = torch.cat((seed, seed2, seed3, seed4), dim=0)
        assert seed.shape == (self.width*batch.shape[0], batch.shape[1], 2)
        return seed
    def cost_func(self, d, pi=None):
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)

    def decomposeAndRevise(self, seeds, reviser_id, action):
        bs, num_nodes, coordinate_dim = seeds.shape  # note that bs = self.batch_size x self.width
        
        cost_ori = (seeds[:, 1:] - seeds[:, :-1]).norm(p=2, dim=2).sum(1) + (seeds[:, 0] - seeds[:, -1]).norm(p=2, dim=1)
        cost_ori, cost_revised_minidx = cost_ori.reshape(-1, self.batch_size).min(0) # width, batch_size
        avg_cost_ori = cost_ori.mean().item()

        revision_len = self.reviser_sizes[reviser_id]
        offset = num_nodes % revision_len
        decomposed_seeds, offset_seed = decomposition(seeds, 
                                        coordinate_dim,
                                        revision_len,
                                        offset,
                                        shift_len=action
                                        )

        original_subtour = torch.arange(0, revision_len, dtype=torch.long).to(device)
        
        decomposed_seeds_revised = revision(self.cost_func, self.revisers[reviser_id], decomposed_seeds, original_subtour)
        
        seeds = decomposed_seeds_revised.reshape(bs, -1, coordinate_dim)

        if offset_seed is not None:
            seeds = torch.cat([seeds,offset_seed], dim=1)

        cost_revised = (seeds[:, 1:] - seeds[:, :-1]).norm(p=2, dim=2).sum(1) + (seeds[:, 0] - seeds[:, -1]).norm(p=2, dim=1)
        cost_revised, cost_revised_minidx = cost_revised.reshape(-1, self.batch_size).min(0) # width, batch_size
        avg_cost_revised = cost_revised.mean().item()
        print('avg_cost_revised:', avg_cost_revised)
        
        return seeds, avg_cost_ori-avg_cost_revised
    
    def e_greedy(self, reviser_id, state):
        if torch.rand(size=(1,)) < self.epsilon:
            action = torch.randint(low=1, high=self.reviser_sizes[reviser_id]//2 + 1, size=(1,))
        else:
            # state starts from 0 in code implementation
            action = torch.argmax(self.q_tables[reviser_id][state]) + 1 # adding one here already
        return action
    
    def learn(self, reviser_id, state, action, reward):
        # state starts from 0
        # action starts from 0
        q_table = self.q_tables[reviser_id]
        q_table[state, action] += self.alpha * (reward+self.gamma*torch.max(q_table[state+1]) - q_table[state, action])
    
    def main_loop(self):
        for iteration in tqdm(range(self.max_iter)):
            print(f'---- iteration {iteration} ----')
            n = self.sampleTSPsize()
            instances = self.sampleTSPInstance(n)
            seeds = self.initializeTours(instances)
            for reviser_id in range(self.M):
                seeds, r = self.decomposeAndRevise(seeds, reviser_id, 0)
                print(reviser_id, r)
                # starts from 0, to I_n-2 
                for state in range(0, self.revision_iters[reviser_id]-1): 
                    action = self.e_greedy(reviser_id, state)
                    seeds, r = self.decomposeAndRevise(seeds, reviser_id, action)
                    # print(reviser_id, r)
                    print(state, action, r)
                    self.learn(reviser_id, state, action-1, r)
        print(self.q_tables)
        for i in range(self.M):
            torch.save(self.q_tables[i], f'./rollers/test_{i}.pt')

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=2, help="Batch size to use training")
    parser.add_argument('--revision_lens', nargs='+', default=[100,50,20] ,type=int)
    parser.add_argument('--revision_iters', nargs='+', default=[20,25,10], type=int)
    parser.add_argument('--decode_strategy', type=str, default='greedy', help='decode strategy of the model')
    parser.add_argument('--width', type=int, default=1, help='Width of tours for an instance')
    parser.add_argument('--epsilon', type=float, default=0.1, help='For epsilon-greedy')
    parser.add_argument('--alpha', type=float, default=0.3, help='Learning rate of rollers')
    parser.add_argument('--gamma', type=float, default=0.9, help='Reward decay')
    parser.add_argument('--max_iter', type=int, default=1, help='The number of iterations for Q-learning')
    opts = parser.parse_args()

    revisers = []
    for reviser_size in opts.revision_lens:
        if reviser_size in [100, 50]:
            reviser_path = f'pretrained_LCP/Reviser-ft/reviser_{reviser_size}/epoch-299.pt'
        
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for reviser in revisers:
        reviser.to(device)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy) # TODO sampling may be better
    q = Q_learning(opts, revisers=revisers)
    q.main_loop()
    


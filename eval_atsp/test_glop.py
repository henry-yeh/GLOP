
"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = True
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
import time
import torch
sys.path.insert(0, '..')
os.chdir(os.path.dirname(os.path.abspath(__file__)))


##########################################################################################
# import

import logging
from utils.insertion import random_insertion_non_euclidean
from utils_atsp.utils import create_logger, copy_all_src
from ATSPTester_glop import ATSPTester as Tester
import numpy as np
##########################################################################################
# parameters

##### GLOP parameters #####
N_REVISER = 50 # We only test on Reviser-50; using more revisers requires code modifications
N_REVISIONS = 3 # number of revision iterations
N_SAMPLES = {
    150: 2000,
    250: 1000,
    1000: 500
    } # for sampling decoding during revision



env_params = {
    'node_cnt': N_REVISER, 
    'problem_gen_params': {
        'int_min': 0,
        'int_max': 1000*1000,
        'scaler': 1000*1000
    },
    'pomo_size': 500,
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256**(1/2),
    'encoder_layer_num': 5,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'softmax', # note here, can be greedy
    'one_hot_seed_cnt': N_REVISER,  # must be >= node_cnt
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/saved_atsp50_model',  # directory path of pre-trained model and log files saved.
        'epoch': 8000,  # epoch version of pre-trained model to load.
    },
    'saved_problem_folder': "../data/n20",
    'saved_problem_filename': 'problem_20_0_1000000_{}.atsp',
    'test_batch_size': 999999, # Note this batch size is for revision
    'augmentation_enable': False, # No augementation for GLOP; requiring code modifications to enable
    'aug_factor': 1,
    'aug_batch_size': 1,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'atsp_matnet_test',
        'filename': 'log.txt'
    }
}



##########################################################################################
# main    

def revision(tour, inst, tester):
    sub_tours = tour.reshape(-1, N_REVISER) # shape: (batch, revision_len)
    sub_insts = [inst[sub_tour][:, sub_tour] for sub_tour in sub_tours]
    original_scores = torch.tensor([cal_len_shpp(sub_tour, inst) for sub_tour in sub_tours]) # note that original_scores are positive values
    # Scale the sub_insts to make the largest value 1
    scale_coef = [sub_inst.max() for sub_inst in sub_insts]
    sub_insts = torch.stack(sub_insts)
    sub_insts_scaled = sub_insts / torch.tensor(scale_coef)[:, None, None]
    
    # Main part of the revision
    revised_scores, solutions = tester.run(sub_insts_scaled) # solutions shape: (batch, revision_len)

    # Scale back the revised scores
    revised_scores = - revised_scores * torch.tensor(scale_coef) # shape: (batch,); add negative sign to make positive value
    
    # TODO: unmcomment to validate the subtours
    for i in range(len(sub_insts)):
        validate_subtour(solutions[i], sub_insts[i], revised_scores[i])

    # Compare the original scores and the revised scores
    improved_scores = original_scores - revised_scores
    # subtours should be aranged in the same order as the original tours, if the improved_scores <= 0
    solutions[improved_scores <= 0] = torch.arange(sub_tours.shape[1])
    # Gather the subtours according to the solutions
    revised_tours = sub_tours.gather(1, solutions)
    # Flatten the revised_tours
    revised_tours = revised_tours.reshape(-1) # shape: (batch * revision_len) i.e. (node_cnt,)
    return revised_tours

def validate_subtour(subtour, dist, cost):
    truth_cost = cal_len_shpp(subtour, dist)
    assert truth_cost - cost < 1e-5
    # Assert subtour is a valid tour: (1) the starting node is 0 and the terminal node is len(subtour) - 1; (2) all nodes are visited exactly once.
    assert subtour[0] == 0 and subtour[-1] == len(subtour) - 1
    for i in range(1, len(subtour) - 1):
        assert i in subtour

def validate_tour(tour):
    for i in range(1, len(tour) - 1):
        assert i in tour

def cal_len(tour, dist):
    cost = dist[tour, torch.roll(tour, -1, -1)].sum()
    return cost.item()

def cal_len_shpp(tour, dist):
    cost = dist[tour[:-1], tour[1:]].sum()
    return cost.item()

def main(n):    
    dataset = torch.load('../data/atsp/ATSP{}.pt'.format(n), map_location='cuda:0')
    env_params['node_cnt'] = 50
    model_params['one_hot_seed_cnt'] = 50
    
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    
    
    torch.random.manual_seed(1)
    order = torch.randperm(n, device='cpu').numpy()
    
    original_costs = []
    revised_costs = []
    # true_cost = []
    
    N_SHIFTS = N_REVISER // N_REVISIONS
    
    start = time.time()
    for inst in dataset:
        tour, cost = random_insertion_non_euclidean(inst, order)
        original_costs.append(cost)
        tour = torch.tensor(tour.astype(np.int64))

        for revision_iter in range(N_REVISIONS):
            tour = revision(tour, inst, tester)
            # Shift the tour to the right by N_SHIFTS
            tour = torch.roll(tour, shifts=N_SHIFTS, dims=-1)

        # TODO: unmcomment to validate the solution
        # validate_tour(tour)
        cost = cal_len(tour, inst)
        revised_costs.append(cost)

    total_duration = time.time() - start
    
    print("insertion costs:   ", sum(original_costs) / len(original_costs))
    print("revised costs:", sum(revised_costs) / len(revised_costs))
    print("total duration: ", total_duration)


if __name__ == "__main__":
    N = int(sys.argv[1])
    env_params['pomo_size'] = N_SAMPLES.get(N, 500)

    main(N)
    
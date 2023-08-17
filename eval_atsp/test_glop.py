
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

env_params = {
    'node_cnt': 50,
    'problem_gen_params': {
        'int_min': 0,
        'int_max': 1000*1000,
        'scaler': 1000*1000
    },
    'pomo_size': 50  # same as node_cnt
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
    'eval_type': 'softmax',
    'one_hot_seed_cnt': 20,  # must be >= node_cnt
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
    'test_batch_size': 1,
    'augmentation_enable': False,
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

L = 1.5

def revision(tour, inst, tester):
    revision_len = env_params['node_cnt']
    assert revision_len == 50
    sub_tours = tour.reshape(-1, revision_len)
    sub_insts = [inst[sub_tour][:, sub_tour] for sub_tour in sub_tours]
    original_scores = torch.stack([inst[sub_tour[:-1], torch.roll(sub_tour, shifts=-1)[:-1]].sum() for sub_tour in sub_tours])
    for sub_inst in sub_insts: # equivalent ATSP of each ASHPP
        sub_inst[:, 0] += L
        sub_inst[:, -1] += L
        sub_inst[0, :] += L
        sub_inst[-1, :] += L
        sub_inst[0, 0] = sub_inst[0, -1] = sub_inst[-1, 0] = sub_inst[-1, -1] = 0
    sub_insts = torch.stack(sub_insts)
    
    revised_scores = torch.stack(tester.run(sub_insts)) - 2 * L
    
    improved = original_scores - revised_scores
    improved[improved < 0] = 0

    return improved.sum().item()
    

def main(n):    
    dataset = torch.load('../data/atsp/ATSP{}.pt'.format(n), map_location='cuda:0')
    env_params['node_cnt'] = 50
    model_params['one_hot_seed_cnt'] = 50
    
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    
    order = torch.randperm(n)
    
    original_costs = []
    revised_costs = []
    
    start = time.time()
    for inst in dataset:
        tour, cost = random_insertion_non_euclidean(inst, order)
        original_costs.append(cost)
        improved_cost = revision(torch.tensor(tour.astype(np.int64)), inst, tester)
        revised_costs.append(cost - improved_cost)
    total_duration = time.time() - start
    
    print("initial costs: ", sum(original_costs) / len(original_costs))
    print("revised costs: ", sum(revised_costs) / len(revised_costs))
    print("total duration: ", total_duration)


if __name__ == "__main__":
    main(int(sys.argv[1]))


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
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))


##########################################################################################
# import

import logging

from utils_atsp.utils import create_logger, copy_all_src
from ATSPTester import ATSPTester as Tester


##########################################################################################
# parameters

env_params = {
    'node_cnt': 20,
    'problem_gen_params': {
        'int_min': 0,
        'int_max': 1000*1000,
        'scaler': 1000*1000
    },
    'pomo_size': 20  # same as node_cnt
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
        'path': './result/saved_atsp100_model',  # directory path of pre-trained model and log files saved.
        'epoch': 12000,  # epoch version of pre-trained model to load.
    },
    'saved_problem_folder': "../data/n20",
    'saved_problem_filename': 'problem_20_0_1000000_{}.atsp',
    'test_batch_size': 1,
    'augmentation_enable': False,
    # 'aug_factor': 1,
    # 'aug_batch_size': 1,
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

def main():

    create_logger(**logger_params)
    _print_config()
    
    n = 250
    dataset = torch.load('../data/atsp/ATSP{}.pt'.format(n), map_location='cuda:0')
    env_params['node_cnt'] = n
    env_params['pomo_size'] = n
    model_params['one_hot_seed_cnt'] = n
    

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params,
                    problems=dataset)

    copy_all_src(tester.result_folder)

    tester.run()



def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()

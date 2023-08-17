
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
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging

from utils.utils import create_logger, copy_all_src
from ATSPTrainer import ATSPTrainer as Trainer


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
    'eval_type': 'argmax',
    'one_hot_seed_cnt': 20,  # must be >= node_cnt
}

optimizer_params = {
    'optimizer': {
        'lr': 4*1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [2001, 2101],  # if further training is needed
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 2000,
    'train_episodes': 10*1000,
    'train_batch_size': 200,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 200,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_atsp_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 2000,  # epoch version of pre-trained model to laod.
    }
}

logger_params = {
    'log_file': {
        'desc': 'matnet_train',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():

    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2
    trainer_params['validate_episodes'] = 4
    trainer_params['validate_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()

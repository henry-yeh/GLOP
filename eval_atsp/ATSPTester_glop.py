
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

import torch

import os
from logging import getLogger

from ASHPPEnv import ASHPPEnv as Env
from ASHPPModel import ASHPPModel as Model

from utils_atsp.utils import get_result_folder, AverageMeter, TimeEstimator

from ATSProblemDef import load_single_problem_from_file


class ATSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = self.tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

        # Load all problems into tensor
        # self.all_problems = problems

    def run(self, insts):

        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = insts.size(0)
        episode = 0
        
        scores = []
        
        solutions = []

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            aug_score, batch_solutions = self._test_one_batch(episode, episode+batch_size, insts)
            
            scores.append(aug_score)
            
            solutions.append(batch_solutions)

            episode += batch_size

            ############################
            # Logs
            ############################
            # elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            # self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
            #     episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            # all_done = (episode == test_num_episode)

            # if all_done:
            #     self.logger.info(" *** Test Done *** ")
            #     self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
            #     self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
        
        scores = torch.cat(scores, dim=0)
        solutions = torch.cat(solutions, dim=0)
        
        return scores, solutions

    def _test_one_batch(self, idx_start, idx_end, insts):

        batch_size = idx_end-idx_start
        problems_batched = insts[idx_start:idx_end]

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            assert False, "Augmentation is not supported"
            aug_factor = self.tester_params['aug_factor']

            batch_size = aug_factor*batch_size
            problems_batched = problems_batched.repeat(aug_factor, 1, 1)
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems_manual(problems_batched)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)

            # Return
            ###############################################
            aug_reward = reward.reshape(batch_size, self.env.pomo_size)
            # shape: (batch, pomo)
            
            # Get solutions
            solutions = self.env.selected_node_list
            # shape: (batch, pomo, node_cnt)
            
            max_pomo_reward, max_pomo_reward_idx = aug_reward.max(dim=1)  # get best results from pomo
            # shape: (batch)

            optimal_solution = solutions[torch.arange(batch_size), max_pomo_reward_idx]
            # shape: (batch, node_cnt)
            
            return max_pomo_reward.float(), optimal_solution  # negative sign to make positive value



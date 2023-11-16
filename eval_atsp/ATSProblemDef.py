
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


def get_random_problems(batch_size, node_cnt, problem_gen_params):

    ################################
    # "tmat" type
    ################################

    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']

    problems = torch.randint(low=int_min, high=int_max, size=(batch_size, node_cnt, node_cnt))
    # shape: (batch, node, node)
    problems[:, torch.arange(node_cnt), torch.arange(node_cnt)] = 0

    while True:
        old_problems = problems.clone()

        problems, _ = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2,3)).min(dim=3)
        # shape: (batch, node, node)

        if (problems == old_problems).all():
            break

    # Scale
    scaled_problems = problems.float() / scaler

    return scaled_problems
    # shape: (batch, node, node)


def load_single_problem_from_file(filename, node_cnt, scaler):

    ################################
    # "tmat" type
    ################################

    problem = torch.empty(size=(node_cnt, node_cnt), dtype=torch.long)
    # shape: (node, node)

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as err:
        print(str(err))

    line_cnt = 0
    for line in lines:
        linedata = line.split()

        if linedata[0].startswith(('TYPE', 'DIMENSION', 'EDGE_WEIGHT_TYPE', 'EDGE_WEIGHT_FORMAT', 'EDGE_WEIGHT_SECTION', 'EOF')):
            continue

        integer_map = map(int, linedata)
        integer_list = list(integer_map)

        problem[line_cnt] = torch.tensor(integer_list, dtype=torch.long)
        line_cnt += 1

    # Diagonals to 0
    problem[torch.arange(node_cnt), torch.arange(node_cnt)] = 0

    # Scale
    scaled_problem = problem.float() / scaler

    return scaled_problem
    # shape: (node, node)


if __name__ == '__main__':
    import os
    problem_gen_params = {
    'int_min': 0,
    'int_max': 1000*1000,
    'scaler': 1000*1000
    }
    # if no such directory, create one
    if not os.path.exists("../data/atsp"):
        os.mkdir("../data/atsp")
    
    torch.manual_seed(1234)   
    
    dataset_size = 30
    for scale in [150, 200, 1000]:
        problems = get_random_problems(dataset_size, scale, problem_gen_params)
        torch.save(problems, "../data/atsp/ATSP{}.pt".format(scale))
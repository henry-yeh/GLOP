import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from utils import load_model
from torch.utils.data import DataLoader
import time
from utils.functions import reconnect, decomposition, load_problem
from problems.tsp.tsp_baseline import solve_insertion
import pprint as pp

mp = torch.multiprocessing.get_context('spawn')

def coordinate_transform(decomposed_seeds):    
    # coordinate transformation
    max_x, indices_max_x = decomposed_seeds[:,:,0].max(dim=1)
    max_y, indices_max_y = decomposed_seeds[:,:,1].max(dim=1)
    min_x, indices_min_x = decomposed_seeds[:,:,0].min(dim=1)
    min_y, indices_min_y = decomposed_seeds[:,:,1].min(dim=1)
    # shapes: (batch_size, ); (batch_size, )
    
    diff_x = max_x - min_x
    diff_y = max_y - min_y
    xy_exchanged = diff_y > diff_x

    # shift to zero
    decomposed_seeds[:, :, 0] -= (min_x).unsqueeze(-1)
    decomposed_seeds[:, :, 1] -= (min_y).unsqueeze(-1)

    # exchange coordinates for those diff_y > diff_x
    decomposed_seeds[xy_exchanged, :, 0], decomposed_seeds[xy_exchanged, :, 1] =  decomposed_seeds[xy_exchanged, :, 1], decomposed_seeds[xy_exchanged, :, 0]
    
    # scale to (0, 1)
    scale_degree = torch.max(diff_x, diff_y)
    scale_degree = scale_degree.view(decomposed_seeds.shape[0], 1, 1)
    decomposed_seeds /= scale_degree

    return decomposed_seeds

def generate(opts):

    pp.pprint(vars(opts))
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")

    
    assert opts.problem_type == 'tsp'
    revision_lens = opts.revision_lens
    assert len(revision_lens) == 1

    reviser_path = opts.load_path
    reviser, _ = load_model(reviser_path, is_local=True)
    revisers = [reviser]

    reviser.to(device)
    reviser.eval()
    reviser.set_decode_type(opts.decode_strategy)

    n = torch.randint(low=200, high=1000, size=(1,))
    
    rg_dataset = None
    while rg_dataset is None or rg_dataset.shape[0] < 1e7:
        dataset = torch.rand(size=(opts.eval_batch_size, n, 2))
        tours = _generate(dataset, opts, device, revisers)

        decomposed_seeds, offset_seeds = decomposition(
            tours, 
            coordinate_dim=2,
            revision_len=opts.problem_size,
            offset=tours.shape[1]%opts.problem_size,
            shift_len=0
            )
        seeds = coordinate_transform(decomposed_seeds)
        # print(seeds.shape)
        # import matplotlib.pyplot as plt
        # plt.scatter(seeds[0,:,0], seeds[0,:,1])
        # plt.show()
        if rg_dataset is None:
            rg_dataset = seeds
        else:
            rg_dataset = torch.cat((rg_dataset, seeds), dim=0)
    return rg_dataset[0: 1000000]


def _generate(batch, opts, device, revisers):

    problem = load_problem(opts.problem_type)

    # cost function for partial solution 
    get_cost_func = lambda input, pi: problem.get_costs(input, pi, return_local=True)

    with torch.no_grad():
        pi_batch = torch.LongTensor(size=batch.shape[:2])
        for instance_id, instance in enumerate(batch):

            cost, pi, duration = solve_insertion(
                                    directory=None, 
                                    name=None, 
                                    loc=instance,
                                    method='farthest',
                                    )
            pi_batch[instance_id] = torch.tensor(pi)
            
        seed = batch.gather(1, pi_batch.unsqueeze(-1).expand_as(batch))
        seed = seed.to(device)
        tours, costs_revised = reconnect( 
                                    get_cost_func=get_cost_func,
                                    batch=seed,
                                    opts=opts,
                                    revisers=revisers,
                                    )
    return tours


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_type", type=str, default='tsp')
    parser.add_argument("--load_path", required=True, type=str)
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--revision_lens', nargs='+', default=[100,] ,type=int)
    parser.add_argument('--revision_iters', nargs='+', default=[10,], type=int)
    parser.add_argument('--shift_lens', nargs='+', default=[10,], type=int)
    parser.add_argument('--decode_strategy', type=str, default='greedy', help='decode strategy of the model')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='Batch size for data generation')
    opts = parser.parse_args()
    
    generate(opts)

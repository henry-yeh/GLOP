import math
import torch
import argparse
import numpy as np
import pprint as pp
import os
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
try:
    from utils import load_model
    from utils.functions import revision, reconnect, decomposition, load_problem
    from problems.tsp.tsp_baseline import solve_insertion
except:
    import sys
    sys.path.insert(0, './')
    from utils import load_model
    from utils.functions import revision, reconnect, decomposition, load_problem
    from problems.tsp.tsp_baseline import solve_insertion

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
    problem = load_problem(opts.problem_type)
    # cost function for partial solution 
    get_cost_func = lambda input, pi: problem.get_costs(input, pi, return_local=True)

    revision_lens = opts.revision_lens
    assert len(revision_lens) == 1

    reviser_path = opts.load_path
    reviser, _ = load_model(reviser_path, is_local=True)
    revisers = [reviser]

    reviser.to(device)
    reviser.eval()
    reviser.set_decode_type(opts.decode_strategy)
    
    FI_dataset = torch.load(opts.data_path, map_location=device)[:500000]
    dataloader = DataLoader(FI_dataset, batch_size=opts.batch_size)

    rg_dataset = None
    for batch in dataloader:
        
        original_subtour = torch.arange(0, opts.revision_lens[0], dtype=torch.long).to(device)
    
        decomposed_seeds_revised, _ = revision(opts, get_cost_func, reviser, batch, original_subtour)

        decomposed_seeds, offset_seeds = decomposition(
            decomposed_seeds_revised, 
            coordinate_dim=2,
            revision_len=opts.tgt_size,
            offset=batch.shape[1]%opts.tgt_size,
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
        print(rg_dataset.shape[0], ' / ', 1000000)

    return rg_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_type", type=str, default='tsp')
    parser.add_argument("--load_path", required=True, type=str)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument('--tgt_size', type=int, default=50)
    parser.add_argument('--revision_lens', nargs='+', default=[100,] ,type=int)
    parser.add_argument('--decode_strategy', type=str, default='greedy', help='decode strategy of the model')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data generation')
    opts = parser.parse_args()
    opts.no_aug = True
    datadir = 'data/RG_train_tsp'
    os.makedirs(datadir, exist_ok=True)
    rg_dataset = generate(opts)
    tgt_size = opts.tgt_size
    torch.save(rg_dataset.cpu(), f'./data/RG_train_tsp/RG{tgt_size}.pt')


import argparse
from logging import warning
import os
import numpy as np
import torch
from utils.data_utils import check_extension, save_dataset
from problems.tsp.tsp_baseline import solve_insertion
from utils.functions import decomposition, revision
from tqdm import tqdm

def solve_and_decompose(instance, opts):
    # instance shape: (problem size, 2); dtype: torch tensor
    problem_size, _ = instance.shape

    cost, pi, duration = solve_insertion(
                            directory=None, 
                            name=None, 
                            loc=instance,
                            method='farthest',
                            )
    
    instance = instance.unsqueeze(0)
    pi = torch.LongTensor(pi)
    seed = instance.gather(1, pi.unsqueeze(-1).expand_as(instance))

    batch_size, num_nodes, coordinate_dim = seed.shape
    
    revision_len = revision_iter = opts.revision_len
    offset = num_nodes % revision_len
    
    dataset = None
    for shift in range(revision_iter):
        decomposed_seeds, offset_seed = decomposition(seed, 
                                                    coordinate_dim,
                                                    revision_len,
                                                    offset,
                                                    shift)
        if dataset is None:
            dataset = decomposed_seeds
        else:
            dataset = torch.cat((dataset, decomposed_seeds), dim=0)
    # print('dataset_ shape:', dataset.shape)
    return dataset 

def generate_tsp_data(dataset_size, opts):
    batch_size = (opts.graph_size//opts.revision_len) * opts.revision_len
    dataset = None
    for _ in tqdm(range(dataset_size//batch_size)):
        instance = torch.rand((opts.graph_size, 2))
        dataset_ = solve_and_decompose(instance, opts)

        # coordinate transformation
        input = dataset_
        max_x, indices_max_x = input[:,:,0].max(dim=1)
        max_y, indices_max_y = input[:,:,1].max(dim=1)
        min_x, indices_min_x = input[:,:,0].min(dim=1)
        min_y, indices_min_y = input[:,:,1].min(dim=1)
        # shapes: (batch_size, ); (batch_size, )
        
        diff_x = max_x - min_x
        diff_y = max_y - min_y
        xy_exchanged = diff_y > diff_x

        # shift to zero
        input[:, :, 0] -= (min_x).unsqueeze(-1)
        input[:, :, 1] -= (min_y).unsqueeze(-1)

        # exchange coordinates for those diff_y > diff_x
        input[xy_exchanged, :, 0], input[xy_exchanged, :, 1] =  input[xy_exchanged, :, 1], input[xy_exchanged, :, 0]
        
        # scale to (0, 1)
        scale_degree = torch.max(diff_x, diff_y)
        scale_degree = scale_degree.view(input.shape[0], 1, 1)
        input /= scale_degree

        if dataset is None:
            dataset = input
        else:
            dataset = torch.cat((dataset, input), dim=0)
    # return shape: (dataset size, problem size, 2); dtype: list
    print('dataset shape:', dataset.shape)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/FI_train_tsp', help="Create datasets in data_dir")
    parser.add_argument("--name", default='FI', type=str, help="Name to identify dataset")
    parser.add_argument("--dataset_size", type=int, default=200000, help="Size of the dataset")
    parser.add_argument('--graph_size', type=int, default=500,
                        help="Sizes of problem instances")
    parser.add_argument('--revision_len', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1235, help="Random seed")
    
    opts = parser.parse_args()

    graph_size = opts.graph_size
    datadir = opts.data_dir
    os.makedirs(datadir, exist_ok=True)
    
    torch.manual_seed(opts.seed)

    filename = os.path.join(datadir, "{}_{}_seed{}{}.pt".format(
    graph_size, opts.name, opts.revision_len, opts.seed))
    dataset = generate_tsp_data(opts.dataset_size, opts)

    torch.save(dataset, filename)

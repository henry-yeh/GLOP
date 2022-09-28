'''
To evaluate a specific reviser on a specific data distribution
'''

import torch
import argparse
import numpy as np
from tqdm import tqdm
from utils import load_model
from utils.functions import load_problem
from train import validate

def eval(opts):
    revisor_path = opts.model_path
    mtl = not opts.disable_mtl
    model, _ = load_model(revisor_path, is_local=True, mtl=mtl)

    # make validation dataset
    problem = load_problem(opts.problem)
    graph_size = opts.graph_size
    filename = f"data/tsp/tsp_{opts.data_distribution}{graph_size}_val_seed1234.pkl"
    val_size = 10000
    val_dataset = problem.make_dataset(
            size=graph_size, num_samples=val_size, filename=filename, distribution=opts.data_distribution)
    avg_reward = validate(model, val_dataset, opts)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='local', help="The problem to solve, default 'local'")
    parser.add_argument('--graph_size', type=int, default=10)
    parser.add_argument('--model_path', type=str, default='pretrained_LCP/Reviser-3-scale-NMT/reviser_10/epoch-199.pt')
    parser.add_argument('--data_distribution', type=str, default='scale')
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--disable_mtl', action='store_true', help='Disable multi-task learning, it is a must when loading a non-MTL model')
    parser.add_argument('--disable-cuda', action='store_true')
    opts = parser.parse_args()
    opts.device = 'cuda' if torch.cuda.is_available() and not opts.disable_cuda else 'cpu'
    eval(opts)

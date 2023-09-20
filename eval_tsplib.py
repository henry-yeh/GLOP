import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from utils import load_model
from torch.utils.data import DataLoader
import time
from utils.functions import reconnect
from utils.functions import load_problem
import pprint as pp
from utils.insertion import random_insertion

torch.manual_seed(1)

def eval_dataset(dataset_path, opts):
    pp.pprint(vars(opts))

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device_id = opts.device_id
    device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")
    print('using device:', device)

    revisers = []
    assert opts.problem_type == 'tsp'
    revision_lens = opts.revision_lens

    for reviser_size in revision_lens:
        reviser_path = f'pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt'
        
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)
        
    for reviser in revisers:
        reviser.to(device)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)

    dataset = reviser.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=0)
    results, duration = _eval_dataset(dataset, opts, device, revisers)

    costs, costs_revised, tours = zip(*results)  # Not really costs since they should be negative
    
    costs = torch.tensor(costs)
    costs_revised = torch.cat(costs_revised, dim=0)
    # tours = torch.cat(tours, dim=0)

    print("Average cost: {} +- {}".format(costs.mean(), (2 * torch.std(costs) / math.sqrt(len(costs))).item()))
    print("Average cost_revised: {} +- {}".format(costs_revised.mean().item(), (2 * torch.std(costs_revised) / math.sqrt(len(costs_revised))).item()))
    print("Total duration: {}".format(duration))

def _eval_dataset(dataset, opts, device, revisers):

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)
    problem = load_problem(opts.problem_type)

    # cost function for partial solution 
    get_cost_func = lambda input, pi: problem.get_costs(input, pi, return_local=True)
    
    results = []

    total_time = 0

    for batch_id, batch in tqdm(enumerate(dataloader), disable=opts.no_progress_bar):
        # tsp batch shape: (bs, problem size, 2)
        avg_cost = 0
        width = opts.width
        problem_size = batch.shape[1]
        print('problem size:', problem_size)

        with torch.no_grad():
            
            if problem_size <= 100:
                width //= 4

            orders = [torch.randperm(problem_size) for i in range(width)]
            
            start = time.time()
            pi_batch = [random_insertion(instance, orders[order_id])[0] for order_id in range(len(orders)) for instance in batch]
            pi_batch = torch.tensor(np.array(pi_batch).astype(np.int64)).reshape(-1, problem_size)
            total_time += time.time() - start

            batch = batch.repeat(width, 1, 1)

            seed = batch.gather(1, pi_batch.unsqueeze(-1).expand_as(batch))
            seed = seed.to(device)

            cost_ori = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
            cost_ori, _ = cost_ori.reshape(-1, opts.eval_batch_size).min(0) # width, bs
            avg_cost = cost_ori.mean().item()
            print('before revision:', avg_cost)

            if problem_size <= 100:
                seed2 = torch.cat((1 - seed[:, :, [0]], seed[:, :, [1]]), dim=2)
                seed3 = torch.cat((seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
                seed4 = torch.cat((1 - seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
                seed = torch.cat((seed, seed2, seed3, seed4), dim=0)
            
            w = seed.shape[0] // opts.eval_batch_size
            print(f'Sample width = {w}')

            if problem_size < 20:
                raise AssertionError
            elif 20 <= problem_size < 50:
                _revisers = revisers[2:]
                opts.no_aug = True
                opts.revision_lens = [20, 10]
                opts.revision_iters = [10, 5] # 2,1
            elif 50<= problem_size < 100:
                opts.no_aug = True
                _revisers = revisers[1:3]
                opts.revision_lens = [50, 20]
                opts.revision_iters = [10, 5] # 2,1
            elif 100<=problem_size< 150:
                opts.no_aug = True
                opts.revision_lens = [100, 50, 20]
                _revisers = revisers[:3]
                opts.revision_iters = [10,5,5] # 4,2,1
            else:
                _revisers = revisers[:3]
                opts.revision_lens = [100, 50, 20]
                opts.revision_iters = [10,10,5] # 4,2,1

            start = time.time()
            tours, costs_revised = reconnect( 
                                        get_cost_func=get_cost_func,
                                        batch=seed,
                                        opts=opts,
                                        revisers=_revisers,
                                        )
            total_time += time.time() - start
            # tours shape: problem_size, 2
            # costs: costs before revision
            # costs_revised: costs after revision

        if costs_revised is not None:
            print('cost_revised:', costs_revised.item())
            results.append((avg_cost, costs_revised, tours))
        else:
            results.append((avg_cost, None, tours))

    return results, total_time


if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", type=int, default=200)
    parser.add_argument("--problem_type", type=str, default='tsp')
    parser.add_argument('--val_size', type=int, default=128,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--revision_lens', nargs='+', default=[100,50,20,10] ,type=int,
                        help='The sizes of revisers')
    parser.add_argument('--revision_iters', nargs='+', default=[20,50,10,], type=int,
                        help='Revision iterations (I_n)')
    parser.add_argument('--decode_strategy', type=str, default='sampling', help='decode strategy of the model')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--width', type=int, default=1,  # 128 / 48
                        help='The initial solutions for a TSP instance generated with diversified insertion')
    parser.add_argument('--no_aug', action='store_true', help='Disable instance augmentation')
    parser.add_argument('--path', type=str, default='', 
                        help='The test dataset path for cross-distribution evaluation')
    parser.add_argument('--no_prune', action='store_true', help='Do not prune the unpromising tours after the first round of revisions')
    opts = parser.parse_args()
    if opts.path == '':
        dataset_path = f'data/tsp/tsp{opts.problem_size}_test.pkl'
    else:
        dataset_path = opts.path
    
    eval_dataset(dataset_path, opts)

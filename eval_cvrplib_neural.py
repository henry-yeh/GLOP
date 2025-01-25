import math
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from utils import load_model
from torch.utils.data import DataLoader
import time
from utils.functions import reconnect
from utils.functions import load_problem
import pprint as pp
from utils.insertion import random_insertion_parallel
from heatmap.cvrp.infer import load_partitioner
from heatmap.cvrp.inst import sum_cost
from problems.cvrp import init  


p_size = {
    "Antwerp1.vrp" : 6000,
    "Antwerp2.vrp" : 7000,
    "Brussels1.vrp" : 15000,
    "Brussels2.vrp" : 16000,
    "Ghent1.vrp" : 10000,
    "Ghent2.vrp" : 11000,
    "Leuven1.vrp" : 3000,
    "Leuven2.vrp" : 4000,
}  

scale = {
    "Antwerp1.vrp" : 1998.0,
    "Antwerp2.vrp" : 1999.0,
    "Brussels1.vrp" : 1982.0,
    "Brussels2.vrp" : 1994.0,
    "Ghent1.vrp" : 1988.0,
    "Ghent2.vrp" : 1996.0,
    "Leuven1.vrp" : 1903.0,
    "Leuven2.vrp" : 1989.0,
}    

optimal = {
    "Antwerp1.vrp" : 477277,
    "Antwerp2.vrp" : 291350,
    "Brussels1.vrp" : 501719,
    "Brussels2.vrp" : 345468,
    "Ghent1.vrp" : 469531,
    "Ghent2.vrp" : 257749,
    "Leuven1.vrp" : 192848,
    "Leuven2.vrp" : 111395,
}


def eval_dataset(dataset_path, opts, partitioner, revisers):
    results, duration = _eval_dataset(dataset_path, opts, opts.device, revisers, partitioner)

    costs, costs_revised, costs_revised_with_penalty, tours = zip(*results)
    costs = torch.tensor(costs)
    costs_revised = torch.stack(costs_revised)

    # print("Average cost: {} +- {}".format(costs.mean(), (2 * torch.std(costs) / math.sqrt(len(costs))).item()))
    # print("Average cost_revised: {} +- {}".format(costs_revised.mean().item(), 
    #                         (2 * torch.std(costs_revised) / math.sqrt(len(costs_revised))).item()))
    # if opts.problem_type == 'pctsp':
    #     print("Average cost_revised with penalty: {} +- {}".format(costs_revised_with_penalty.mean().item(), 
    #                         (2 * torch.std(costs_revised_with_penalty) / math.sqrt(len(costs_revised_with_penalty))).item()))
    
    # if opts.problem_type != 'cvrp':
    #     tours = torch.cat(tours, dim=0)
    return costs_revised, duration

def _eval_dataset(dataset_path, opts, device, revisers, partitioner):
    total_time = 0
    start = time.time()
    dataset, n_tsps_per_route_lst = init(dataset_path, opts, partitioner)
    
    if dataset[0].shape[1] < 50:
        revisers = revisers[1:]
        opts.revision_lens = opts.revision_lens[1:]
        opts.revision_ites = opts.revision_iters[1:]
    
    total_time += time.time() - start
        
    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    problem = load_problem('tsp')
    get_cost_func = lambda input, pi: problem.get_costs(input, pi, return_local=True)
    
    results = []
    
    
    for batch_id, batch in tqdm(enumerate(dataloader), disable=opts.no_progress_bar):
        # tsp batch shape: (bs, problem size, 2)
        avg_cost = 0
        start = time.time()
        with torch.no_grad():
            batch = batch.squeeze() # (n_subTSPs_for_width_routes, max_seq_len, 2)
            n_subTSPs, max_seq_len, _ = batch.shape
            n_tsps_per_route = n_tsps_per_route_lst[batch_id]
            assert sum(n_tsps_per_route) == n_subTSPs
            opts.eval_batch_size = n_subTSPs
            order = torch.arange(max_seq_len)
            pi_batch = random_insertion_parallel(batch, order)
            pi_batch = torch.tensor(pi_batch.astype(np.int64))
            assert pi_batch.shape == (n_subTSPs, max_seq_len)
            seed = batch.gather(1, pi_batch.unsqueeze(-1).repeat(1,1,2))
            assert seed.shape == (n_subTSPs, max_seq_len, 2)

                
            seed = seed.to(device)
            cost_ori = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
            avg_cost = sum_cost(cost_ori, n_tsps_per_route).min()


                
            tours, costs_revised = reconnect( 
                                        get_cost_func=get_cost_func,
                                        batch=seed,
                                        opts=opts,
                                        revisers=revisers,
                                        )
        total_time += time.time() - start

        assert costs_revised.shape == (n_subTSPs,)
        costs_revised, best_partition_idx = sum_cost(costs_revised, n_tsps_per_route).min(dim=0)
        subtour_start = sum(n_tsps_per_route[:best_partition_idx])
        tours = tours[subtour_start: subtour_start+n_tsps_per_route[best_partition_idx]]
        assert tours.shape == (n_tsps_per_route[best_partition_idx], max_seq_len, 2)
        tours = tours.reshape(-1, 2)

        results.append((avg_cost, costs_revised, None, tours))

    return results, total_time             
            
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    # parser.add_argument("--problem_size", type=int, default='')
    parser.add_argument("--problem_type", type=str, default='cvrplib')
    parser.add_argument('--val_size', type=int, default=1,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--revision_lens', nargs='+', default=[50, 20] ,type=int,
                        help='The sizes of revisers')
    parser.add_argument('--revision_iters', nargs='+', default=[5, 5], type=int,
                        help='Revision iterations (I_n)')
    parser.add_argument('--decode_strategy', type=str, default='greedy', help='decode strategy of the model')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--width', type=int, default=1, 
                        help='The initial solutions for a TSP instance generated with diversified insertion')
    parser.add_argument('--no_aug', action='store_true', help='Disable instance augmentation')
    parser.add_argument('--path', type=str, default='', 
                        help='The test dataset path for cross-distribution evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--n_subset', type=int, default=1, help='The number of stochastically constructed PCTSP node subsets')
    parser.add_argument('--n_partition', type=int, default=1, help='The number of stochastically constructed CVRP partitions')
    parser.add_argument('--ckpt_path', type=str, default='', help='Checkpoint path for CVRP eval')
    parser.add_argument('--no_prune', action='store_true', help='Do not prune the unpromising tours after the first round of revisions')
    opts = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device_id = opts.device_id
    device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")
    opts.device = device
    print('using device:', device)

    torch.manual_seed(opts.seed)
    
    ckpt_path = "./pretrained/Partitioner/cvrp/cvrp-2000-cvrplib.pt" if opts.ckpt_path == '' else opts.ckpt_path   
    partitioner = load_partitioner(2000, opts.device, ckpt_path, 300, 6)
    
    revisers = []
    for reviser_size in opts.revision_lens:
        reviser_path = f'pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt'
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)
        
    for reviser in revisers:
        reviser.to(opts.device)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)
        
    
    for name in scale.keys():
        opts.revision_lens = [50, 20]
        opts.revision_iters = [5, 5] 
        opts.probelm_size = p_size[name]
        path = 'data/vrp/cvrplib/' + name + ".pkl"
        cost, durarion = eval_dataset(path, opts, partitioner, revisers)
        scale_fac = scale[name]
        optimal_obj = optimal[name]
        gap = cost * scale_fac / optimal_obj - 1
        print(name, "- Opt. gap: ", gap.item())
        print('Duration: ', durarion)
        print()
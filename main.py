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
from utils.insertion import random_insertion
from heatmap.cvrp.infer import load_partitioner
from heatmap.cvrp.inst import sum_cost


def eval_dataset(dataset_path, opts):
    pp.pprint(vars(opts))
    
    revisers = []
    revision_lens = opts.revision_lens

    for reviser_size in revision_lens:
        reviser_path = f'pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt'
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)
        
    for reviser in revisers:
        reviser.to(opts.device)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)
        
    results, duration = _eval_dataset(dataset_path, opts, opts.device, revisers)

    costs, costs_revised, costs_revised_with_penalty, tours = zip(*results)
    costs = torch.tensor(costs)
    if opts.problem_type in ['cvrp', 'cvrplib']:
        costs_revised = torch.stack(costs_revised)
    else:
        costs_revised = torch.cat(costs_revised, dim=0)
    
    if opts.problem_type == 'pctsp':
        costs_revised_with_penalty = torch.cat(costs_revised_with_penalty, dim=0)

    print("Average cost: {} +- {}".format(costs.mean(), (2 * torch.std(costs) / math.sqrt(len(costs))).item()))
    print("Average cost_revised: {} +- {}".format(costs_revised.mean().item(), 
                            (2 * torch.std(costs_revised) / math.sqrt(len(costs_revised))).item()))
    if opts.problem_type == 'pctsp':
        print("Average cost_revised with penalty: {} +- {}".format(costs_revised_with_penalty.mean().item(), 
                            (2 * torch.std(costs_revised_with_penalty) / math.sqrt(len(costs_revised_with_penalty))).item()))
    print("Total duration: {}".format(duration))
    
    if opts.problem_type != 'cvrp':
        tours = torch.cat(tours, dim=0)
    return tours

def _eval_dataset(dataset_path, opts, device, revisers):
    start = time.time()
    if opts.problem_type == 'tsp':
        dataset = revisers[0].problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=0)
        if opts.problem_size <= 100:
            if opts.width >= 4:
                opts.width //= 4
                opts.tsp_aug = True
            else:
                opts.tsp_aug = False
        orders = [torch.randperm(opts.problem_size) for i in range(opts.width)]
        pi_all = [random_insertion(instance, orders[order_id])[0] for order_id in range(len(orders)) for instance in dataset] # instance: (p_size, 2)
        pi_all = torch.tensor(np.array(pi_all).astype(np.int64)).reshape(len(orders), opts.val_size, opts.problem_size) # width, val_size, p_size
    elif opts.problem_type == 'pctsp': # dataset (n_cons*val_size, p_size, 2), pi_all (width, n_cons*val_size, p_size), penalty (n_cons, val_size)
        from problems.pctsp import init
        opts.eval_batch_size = opts.eval_batch_size * opts.n_subset
        dataset, penalty= init(dataset_path, opts) # (n_val*n_subset, max_seq_len, 2), (val_size*n_subset, )
        dataset = dataset.cpu()
        max_seq_len = dataset.size(1)
        order = torch.arange(max_seq_len) # width=1 by default for pctsp
        pi_all = [random_insertion(instance, order)[0] for instance in dataset]
        pi_all = torch.tensor(np.array(pi_all).astype(np.int64)).unsqueeze(0)  # (1, n_val*n_subset, max_seq_len)
        assert pi_all.shape == (1, opts.val_size*opts.n_subset, max_seq_len)
    elif opts.problem_type == 'cvrp':
        from problems.cvrp import init  
        dataset, n_tsps_per_route_lst = init(dataset_path, opts)
        opts.eval_batch_size = 1
    elif opts.problem_type == 'cvrplib':
        from problems.cvrp import init  
        ckpt_path = "./pretrained/Partitioner/cvrp/cvrp-2000-cvrplib.pt" if opts.ckpt_path == '' else opts.ckpt_path   
        partitioner = load_partitioner(2000, opts.device, ckpt_path, 300, 6)
        dataset, n_tsps_per_route_lst = init(dataset_path, opts, partitioner)
        opts.eval_batch_size = 1
        
    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)
    

    problem = load_problem('tsp')
    get_cost_func = lambda input, pi: problem.get_costs(input, pi, return_local=True)
    
    results = []
    for batch_id, batch in tqdm(enumerate(dataloader), disable=opts.no_progress_bar):
        # tsp batch shape: (bs, problem size, 2)
        avg_cost = 0
        with torch.no_grad():
            if opts.problem_type in ['tsp', 'pctsp']:
                p_size = batch.size(1)
                batch = batch.repeat(opts.width, 1, 1) # (1,1,1) for pctsp
                pi_batch = pi_all[:, batch_id*opts.eval_batch_size: (batch_id+1)*opts.eval_batch_size, :].reshape(-1, p_size)
                seed = batch.gather(1, pi_batch.unsqueeze(-1).repeat(1,1,2))
            elif opts.problem_type in ['cvrp', 'cvrplib']:
                batch = batch.squeeze() # (n_subTSPs_for_width_routes, max_seq_len, 2)
                n_subTSPs, max_seq_len, _ = batch.shape
                n_tsps_per_route = n_tsps_per_route_lst[batch_id]
                assert sum(n_tsps_per_route) == n_subTSPs
                opts.eval_batch_size = n_subTSPs
                order = torch.arange(max_seq_len)
                pi_batch = [random_insertion(instance, order)[0] for instance in batch]
                pi_batch = torch.tensor(np.array(pi_batch).astype(np.int64))
                assert pi_batch.shape == (n_subTSPs, max_seq_len)
                seed = batch.gather(1, pi_batch.unsqueeze(-1).repeat(1,1,2))
                assert seed.shape == (n_subTSPs, max_seq_len, 2)
            else:
                raise NotImplementedError
                
            seed = seed.to(device)
            cost_ori = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
            if opts.problem_type in ['tsp', 'pctsp']:
                cost_ori, _ = cost_ori.reshape(-1, opts.eval_batch_size).min(0) # width, bs
                avg_cost = cost_ori.mean().item()
            elif opts.problem_type in ['cvrp', 'cvrplib']:
                avg_cost = sum_cost(cost_ori, n_tsps_per_route).min()
            else:
                raise NotImplementedError

            if opts.problem_size <= 100 and opts.problem_type=='tsp' and opts.tsp_aug:
                seed2 = torch.cat((1 - seed[:, :, [0]], seed[:, :, [1]]), dim=2)
                seed3 = torch.cat((seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
                seed4 = torch.cat((1 - seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
                seed = torch.cat((seed, seed2, seed3, seed4), dim=0)
                
            tours, costs_revised = reconnect( 
                                        get_cost_func=get_cost_func,
                                        batch=seed,
                                        opts=opts,
                                        revisers=revisers,
                                        )

        if opts.problem_type == 'pctsp':
            costs_revised_with_penalty, costs_revised_minidx = (costs_revised.reshape(-1, opts.n_subset)+ \
                penalty[batch_id*opts.eval_batch_size: (batch_id+1)*opts.eval_batch_size].reshape(-1, opts.n_subset)).min(1)
            costs_revised, _ = costs_revised.reshape(-1, opts.n_subset).min(1)
            tours = tours.reshape(-1, opts.n_subset, max_seq_len, 2)[torch.arange(opts.eval_batch_size//opts.n_subset), costs_revised_minidx, :, :]
            assert costs_revised.size(0) == costs_revised_with_penalty.size(0) == tours.size(0) == opts.eval_batch_size//opts.n_subset
        elif opts.problem_type in ['cvrp', 'cvrplib']:
            assert costs_revised.shape == (n_subTSPs,)
            costs_revised, best_partition_idx = sum_cost(costs_revised, n_tsps_per_route).min(dim=0)
            subtour_start = sum(n_tsps_per_route[:best_partition_idx])
            tours = tours[subtour_start: subtour_start+n_tsps_per_route[best_partition_idx]]
            assert tours.shape == (n_tsps_per_route[best_partition_idx], max_seq_len, 2)
            tours = tours.reshape(-1, 2)
        
        if opts.problem_type == 'pctsp':
            results.append((avg_cost, costs_revised, costs_revised_with_penalty, tours))
        elif opts.problem_type in ['tsp', 'cvrp', 'cvrplib']:
            results.append((avg_cost, costs_revised, None, tours))
        else:
            raise NotImplementedError
        

    duration = time.time() - start

    return results, duration               
            
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", type=int, default=200)
    parser.add_argument("--problem_type", type=str, default='tsp')
    parser.add_argument('--val_size', type=int, default=128,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--revision_lens', nargs='+', default=[20] ,type=int,
                        help='The sizes of revisers')
    parser.add_argument('--revision_iters', nargs='+', default=[10,], type=int,
                        help='Revision iterations (I_n)')
    parser.add_argument('--decode_strategy', type=str, default='sampling', help='decode strategy of the model')
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

    if opts.path == '':
        if opts.problem_type == 'tsp':
            opts.path = f'data/tsp/tsp{opts.problem_size}_test.pkl'
        elif opts.problem_type == 'cvrp':
            opts.path = f'data/vrp/vrp{opts.problem_size}_test_seed1234.pkl'
        elif opts.problem_type == 'pctsp':
            opts.path = f'data/pctsp/pctsp{opts.problem_size}_test_seed1234.pkl'
        else:
            raise NotImplementedError
        
    if opts.problem_type == 'cvrp':
        if opts.eval_batch_size != 1:
            opts.eval_batch_size = 1
            warnings.warn('Set eval_batch_size to 1 for CVRP!')
        if opts.width != 1:
            opts.width = 1
            warnings.warn('Set width to 1 for CVRP!')
    if opts.problem_type == 'pctsp':
        if opts.width != 1:
            opts.width = 1
            warnings.warn('Set width to 1 for PCTSP!')
        
    torch.manual_seed(opts.seed)
        
    tours = eval_dataset(opts.path, opts)
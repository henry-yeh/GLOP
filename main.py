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



def eval_dataset(dataset_path, opts):
    pp.pprint(vars(opts))

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device_id = opts.device_id
    device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")
    opts.device = device
    print('using device:', device)

    revisers = []
    revision_lens = opts.revision_lens

    for reviser_size in revision_lens:
        reviser_path = f'pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt'
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)
        
    for reviser in revisers:
        reviser.to(device)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)
        
    results, duration = _eval_dataset(dataset_path, opts, device, revisers)

    costs, costs_revised, costs_revised_with_penalty, tours = zip(*results)  # Not really costs since they should be negative
    
    costs = torch.tensor(costs)
    if opts.problem_type == 'cvrp':
        costs_revised = torch.stack(costs_revised)
    else:
        costs_revised = torch.cat(costs_revised, dim=0)
    if opts.problem_type == 'pctsp' or opts.problem_type == 'spctsp':
        costs_revised_with_penalty = torch.cat(costs_revised_with_penalty, dim=0)

    print("Average cost: {} +- {}".format(costs.mean(), (2 * torch.std(costs) / math.sqrt(len(costs))).item()))
    print("Average cost_revised: {} +- {}".format(costs_revised.mean().item(), 
                            (2 * torch.std(costs_revised) / math.sqrt(len(costs_revised))).item()))
    if opts.problem_type == 'pctsp' or opts.problem_type == 'spctsp':
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
    elif opts.problem_type in ['pctsp', 'spctsp']: # dataset (n_cons*val_size, p_size, 2), pi_all (width, n_cons*val_size, p_size), penalty (n_cons, val_size)
        from problems.pctsp import init
        n_cons = opts.n_subset
        opts.eval_batch_size = opts.eval_batch_size * n_cons
        dataset, penalty= init(dataset_path, opts)
        penalty = penalty.contiguous().reshape(-1)
        dataset = dataset.cpu()
        orders = [torch.randperm(opts.problem_size+1) for i in range(opts.width)]
        pi_all = [random_insertion(instance, orders[order_id])[0] for order_id in range(len(orders)) for instance in dataset]
        pi_all = torch.tensor(np.array(pi_all).astype(np.int64)).reshape(len(orders), opts.val_size*n_cons, opts.problem_size+1)
    elif opts.problem_type == 'cvrp':
        from utils.insertion import cvrplib_random_insertion
        from problems.cvrp import init  
        dataset = init(opts, cvrplib_random_insertion)
        print('subTSP size:', dataset[0].size(1))
    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)
    print('Total insertion time:', time.time()-start)
    # print(pi_all.shape)
    # print(dataset)
    

    problem = load_problem('tsp')
    get_cost_func = lambda input, pi: problem.get_costs(input, pi, return_local=True)
    
    results = []
    for batch_id, batch in tqdm(enumerate(dataloader), disable=opts.no_progress_bar):
        # tsp batch shape: (bs, problem size, 2)
        avg_cost = 0
        with torch.no_grad():
            if opts.problem_type != 'cvrp':
                p_size = batch.size(1)
                batch = batch.repeat(opts.width, 1, 1)
                pi_batch = pi_all[:, batch_id*opts.eval_batch_size: (batch_id+1)*opts.eval_batch_size, :].reshape(-1, p_size)
                seed = batch.gather(1, pi_batch.unsqueeze(-1).expand_as(batch))
            else:
                seed = batch.squeeze()
                # print('batch shape',seed.shape)
            seed = seed.to(device)
            cost_ori = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
            if opts.problem_type != 'cvrp':
                cost_ori, _ = cost_ori.reshape(-1, opts.eval_batch_size).min(0) # width, bs
                avg_cost = cost_ori.mean().item()
            else:
                avg_cost = cost_ori.sum().item()
            if opts.problem_size <= 100 and opts.tsp_aug and opts.problem_type=='tsp':
                seed2 = torch.cat((1 - seed[:, :, [0]], seed[:, :, [1]]), dim=2)
                seed3 = torch.cat((seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
                seed4 = torch.cat((1 - seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
                seed = torch.cat((seed, seed2, seed3, seed4), dim=0)
            w = seed.shape[0] // opts.eval_batch_size
            tours, costs_revised = reconnect( 
                                        get_cost_func=get_cost_func,
                                        batch=seed,
                                        opts=opts,
                                        revisers=revisers,
                                        )
            # tours shape: problem_size, 2
            # costs: costs before revision
            # costs_revised: costs after revision
        if opts.problem_type == 'pctsp' or opts.problem_type == 'spctsp':
            costs_revised_with_penalty, costs_revised_minidx = (costs_revised.reshape(n_cons, -1)+penalty[batch_id*opts.eval_batch_size: (batch_id+1)*opts.eval_batch_size].reshape(n_cons, -1)).min(0) # n_cons, bs
            costs_revised, _ = costs_revised.reshape(n_cons, -1).min(0) # n_cons, bs
            tours = tours.reshape(n_cons, -1, opts.problem_size+1, 2)[costs_revised_minidx, torch.arange(opts.eval_batch_size//n_cons), :, :]
        elif opts.problem_type == 'cvrp':
            costs_revised = costs_revised.sum()
            tours = tours.reshape(-1, 2)
        if opts.problem_type in ['pctsp', 'spctsp']:
            results.append((avg_cost, costs_revised, costs_revised_with_penalty, tours))
        elif opts.problem_type in ['tsp', 'cvrp']:
            results.append((avg_cost, costs_revised, None, tours))

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
    parser.add_argument('--revision_lens', nargs='+', default=[100,50,20] ,type=int,
                        help='The sizes of revisers')
    parser.add_argument('--revision_iters', nargs='+', default=[20,50,10,], type=int,
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
    
    opts = parser.parse_args()
    
    if opts.path == '':
        if opts.problem_type == 'tsp':
            opts.path = f'data/tsp/tsp{opts.problem_size}_test.pkl'
        elif opts.problem_type == 'cvrp':
            opts.path = f'data/vrp/vrp{opts.problem_size}_test_seed1234.pkl'
        else:
            opts.path = f'data/{opts.problem_type}/{opts.problem_type}{opts.problem_size}_test_seed1234.pkl'
    dataset_path = opts.path
        
    if opts.problem_type == 'cvrp':
        if opts.eval_batch_size != 1:
            opts.eval_batch_size = 1
            warnings.warn('Set eval_batch_size to 1 for CVRP!')
        if opts.width != 1:
            opts.width = 1
            warnings.warn('Set width to 1 for CVRP!')
        
    torch.manual_seed(opts.seed)
        
    tours = eval_dataset(dataset_path, opts)
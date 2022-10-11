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
from problems.tsp.tsp_baseline import solve_insertion
import pprint as pp

mp = torch.multiprocessing.get_context('spawn')


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset(dataset_path, opts):
    pp.pprint(vars(opts))
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")

    revisers = []
    assert opts.problem_type == 'tsp'
    revision_lens = opts.revision_lens

    for reviser_size in revision_lens:
        reviser_path = f'pretrained_LCP/reviser_{reviser_size}/mean-sum/epoch-100.pt'
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)
        
    for reviser in revisers:
        reviser.to(device)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy) # TODO sampling may be better

    dataset = reviser.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=0)
    results = _eval_dataset(dataset, opts, device, revisers)

    costs, costs_revised, tours, durations = zip(*results)  # Not really costs since they should be negative
    
    costs = torch.tensor(costs)
    costs_revised = torch.cat(costs_revised, dim=0)
    tours = torch.cat(tours, dim=0)

    print("Average cost: {} +- {}".format(costs.mean(), (2 * torch.std(costs) / math.sqrt(len(costs))).item()))
    print("Average cost_revised: {} +- {}".format(costs_revised.mean().item(), (2 * torch.std(costs_revised) / math.sqrt(len(costs_revised))).item()))
    print("Total duration: {}".format(sum(durations)))
    print("Avg. duration: {}".format(sum(durations) / opts.val_size))

    return costs, tours, durations


def _eval_dataset(dataset, opts, device, revisers):

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)
    problem = load_problem(opts.problem_type)

    # cost function for partial solution 
    get_cost_func = lambda input, pi: problem.get_costs(input, pi, return_local=True)

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        # tsp batch shape: (bs, problem size, 2)
        avg_cost = 0
        start = time.time()
        with torch.no_grad():
                
            if opts.FI:
                pi_batch = torch.LongTensor(size=(opts.eval_batch_size, opts.problem_size))
                for instance_id, instance in enumerate(batch):

                    cost, pi, duration = solve_insertion(
                                            directory=None, 
                                            name=None, 
                                            loc=instance,
                                            method='farthest',
                                            )
                    avg_cost += cost
                    pi_batch[instance_id] = torch.tensor(pi)
                avg_cost /= opts.eval_batch_size

            else:
                pi_batch = torch.stack([torch.randperm(opts.problem_size) for _ in range(opts.width*opts.eval_batch_size)])
                batch = batch.repeat(opts.width, 1, 1)
                
            seed = batch.gather(1, pi_batch.unsqueeze(-1).expand_as(batch))
            seed = seed.to(device)

            if not opts.FI:
                avg_cost = ((seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)).mean()

            ##############################################################
            # if opts.aug:
                
            #     if opts.aug_shift > 1:
            #         seed = torch.cat([torch.roll(seed, i, 1) for i in range(0, opts.aug_shift)], dim=0)
            #     seed2 = torch.cat((1 - seed[:, :, [0]], seed[:, :, [1]]), dim=2)
            #     seed3 = torch.cat((seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
            #     seed4 = torch.cat((1 - seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
            #     seed = torch.cat((seed, seed2, seed3, seed4), dim=0)
                
            #     print('\n seed shape after augmentation:', seed.shape)

            # needed shape: (width, graph_size, 2) / (width, graph_size)
            
            tours, costs_revised = reconnect( 
                                        get_cost_func=get_cost_func,
                                        batch=seed,
                                        opts=opts,
                                        revisers=revisers,
                                        )
            # tours shape: problem_size, 2
            # costs: costs before revision
            # costs_revised: costs after revision

        duration = time.time() - start

        if costs_revised is not None:
            results.append((avg_cost, costs_revised, tours, duration))
        else:
            results.append((avg_cost, None, tours, duration))

    return results


if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", type=int, default=200)
    parser.add_argument("--problem_type", type=str, default='tsp')
    parser.add_argument('--val_size', type=int, default=16,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--revision_lens', nargs='+', default=[20] ,type=int)
    parser.add_argument('--revision_iters', nargs='+', default=[20], type=int)
    parser.add_argument('--shift_lens', nargs='+', default=[1], type=int)
    parser.add_argument('--decode_strategy', type=str, default='greedy', help='decode strategy of the model')
    parser.add_argument('--width', type=int, default=1, help='number of candidate solutions / seeds (M)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--FI', action='store_true')
    
    opts = parser.parse_args()


    dataset_path = f'data/tsp/tsp{opts.problem_size}_test.pkl'
    eval_dataset(dataset_path, opts)

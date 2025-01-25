import torch
import numpy as np
from utils.functions import load_problem, reconnect
from utils.insertion import random_insertion_parallel

problem = load_problem('tsp')
get_cost_func = lambda input, pi: problem.get_costs(input, pi, return_local=True)

@torch.no_grad()
def eval(subsets, coor, sol_penalty, opts): # eval_batch_size, revision_len, revision_iters, no_aug
    p_size = subsets.size(1)
    seeds = coor[subsets] # (width, p_size, 2)
    order = torch.arange(p_size)
    pi_all = random_insertion_parallel(seeds, order)
    pi_all = torch.tensor(pi_all.astype(np.int64), device=seeds.device).reshape(-1, p_size) # width, p_size
    seeds = seeds.gather(1, pi_all.unsqueeze(-1).expand_as(seeds))
    tours, costs_revised = reconnect( 
                                get_cost_func=get_cost_func,
                                batch=seeds,
                                opts=opts,
                                revisers=opts.revisers,
                                )
    return costs_revised + sol_penalty
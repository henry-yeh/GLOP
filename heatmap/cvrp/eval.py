import torch
import numpy as np
from utils.functions import load_problem, reconnect
from utils.insertion import random_insertion_parallel
from heatmap.cvrp.inst import sum_cost

problem = load_problem('tsp')
get_cost_func = lambda input, pi: problem.get_costs(input, pi, return_local=True)

@torch.no_grad()
def eval(tsp_insts, n_tsps_per_route, opts):
    opts.eval_batch_size = (tsp_insts.size(0))
    p_size = tsp_insts.size(1)
    seeds = tsp_insts
    order = torch.arange(p_size)
    pi_all = random_insertion_parallel(seeds, order)
    pi_all = torch.tensor(pi_all.astype(np.int64), device=seeds.device).reshape(-1, p_size)
    seeds = seeds.gather(1, pi_all.unsqueeze(-1).expand_as(seeds))
    tours, costs_revised = reconnect( 
                                get_cost_func=get_cost_func,
                                batch=seeds,
                                opts=opts,
                                revisers=opts.revisers,
                                )
    assert costs_revised.size(0) == seeds.size(0)
    costs_revised = sum_cost(costs_revised, n_tsps_per_route)
    return costs_revised
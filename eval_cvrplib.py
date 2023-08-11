import time
import argparse
import torch
import pprint as pp

from problems.cvrp import init  
from heatmap.cvrp.inst import sum_cost
from utils.lkh import lkh_solve
from heatmap.cvrp.infer import load_partitioner


def cvrp_lkh_eval(path, opts, partitioner):
    start = time.time()
    dataset, n_tsps_per_route_lst = init(path, opts, partitioner)
    sum_time = time.time() - start
    subtsps = dataset[0].numpy().tolist()
    costs, duration = lkh_solve(opts, subtsps, 0)
    costs = sum_cost(costs, n_tsps_per_route_lst[0])
    sum_time += duration
    print('Total duration:', sum_time)
    return costs.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--problem_size', type=int, default=1000, help='Problem size of CVRP')
    parser.add_argument('--val_size', type=int, default=100, help='Number of instances used for reporting test performance')
    parser.add_argument('--n_partition', type=int, default=1, help='The number of stochastically constructed CVRP partitions')
    parser.add_argument("--cpus", type=int, default=12, help="Number of CPUs to use")
    parser.add_argument('--disable_cache', action='store_true', help='Disable caching')
    parser.add_argument('--max_calc_batch_size', type=int, default=1000, help='Size for subbatches')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, help="Number of instances to process")
    parser.add_argument('--offset', type=int, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--path', type=str, default='')

    opts = parser.parse_args()
    
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device_id = opts.device_id
    device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")
    opts.device = device
    
    torch.manual_seed(opts.seed)

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
    
    ckpt_path = "./pretrained/Partitioner/cvrp/cvrp-2000-cvrplib.pt" if opts.ckpt_path == '' else opts.ckpt_path   
    partitioner = load_partitioner(2000, opts.device, ckpt_path, 300, 6)
    
    pp.pprint(vars(opts))
    
    gaps = []
    for name in scale.keys():
        filename = 'data/cvrplib/' + name + ".pkl"
        opts.problem_size = p_size[name]
        opts.val_size = 1
        scale_fac = scale[name]
        optimal_obj = optimal[name]
        cost = cvrp_lkh_eval(filename, opts, partitioner)
        gap = cost * scale_fac / optimal_obj - 1
        print(name, "- Opt. gap: ", gap)
        gaps.append(gap)
    print('Avg. gap:', sum(gaps)/len(gaps))
    
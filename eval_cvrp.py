import time
import argparse
import torch
import pprint as pp

from problems.cvrp import init  
from heatmap.cvrp.inst import sum_cost
from utils.lkh import lkh_solve


def cvrp_lkh_eval(path, opts):
    pp.pprint(vars(opts))
    
    start = time.time()
    dataset, n_tsps_per_route_lst = init(dataset_path, opts)
    
    sum_time = time.time() - start
    avg_time_4_heatmap = sum_time / opts.val_size
    
    cost_lst = []
    time_lst = []
    for instance_id in range(opts.val_size):
        subtsps = dataset[instance_id].numpy().tolist()
        costs, duration = lkh_solve(opts, subtsps, instance_id)
        costs = sum_cost(costs, n_tsps_per_route_lst[instance_id])
        assert costs.shape == (opts.n_partition,)
        cost = costs.min()
        total_time = duration + avg_time_4_heatmap
        cost_lst.append(cost)
        time_lst.append(total_time)
        print('cost', cost.item(), '| time', total_time)
        
    print('Average cost:', torch.stack(cost_lst).mean().item())
    print('Average duration:', torch.tensor(time_lst).mean().item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--problem_size', type=int, default=1000, help='Problem size of CVRP')
    parser.add_argument('--val_size', type=int, default=100, help='Number of instances used for reporting test performance')
    parser.add_argument('--n_partition', type=int, default=1, help='The number of stochastically constructed CVRP partitions')
    parser.add_argument("--cpus", type=int, required=True, help="Number of CPUs to use")
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
    
    if opts.path == '':
        dataset_path = f'data/vrp/vrp{opts.problem_size}_test_seed1234.pkl'
    else:
        dataset_path = opts.path
        
    cvrp_lkh_eval(dataset_path, opts)
    
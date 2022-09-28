import warnings

import torch
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F
import math

def load_problem(name):
    from problems import TSP, CVRP, SDVRP, OP, PCTSPDet, PCTSPStoch,LOCAL
    problem = {
        'local': LOCAL,
        'tsp': TSP,
        'cvrp': CVRP,
        'sdvrp': SDVRP,
        'op': OP,
        'pctsp_det': PCTSPDet,
        'pctsp_stoch': PCTSPStoch,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem

def solve_gurobi(directory, name, loc, disable_cache=False, timeout=None, gap=None):
    # Lazy import so we do not need to have gurobi installed to run this script
    from problems.tsp.tsp_gurobi import solve_euclidian_tsp as solve_euclidian_tsp_gurobi

    try:
        problem_filename = os.path.join(directory, "{}.gurobi{}{}.pkl".format(
            name, "" if timeout is None else "t{}".format(timeout), "" if gap is None else "gap{}".format(gap)))

        if os.path.isfile(problem_filename) and not disable_cache:
            (cost, tour, duration) = load_dataset(problem_filename)
        else:
            # 0 = start, 1 = end so add depot twice
            start = time.time()
            
            loc.append(loc[int(len(loc)-1)])
            
            cost, tour = solve_euclidian_tsp_gurobi(loc, threads=1, timeout=timeout, gap=gap)
            duration = time.time() - start  # Measure clock time
            
#             save_dataset((cost, tour, duration), problem_filename)

        # First and last node are depot(s), so first node is 2 but should be 1 (as depot is 0) so subtract 1
        total_cost = calc_tsp_length(loc, tour)
        print(total_cost)
        
#         assert abs(total_cost - cost) <= 1e-5, "Cost is incorrect"
        return total_cost, tour, duration

    except Exception as e:
        # For some stupid reason, sometimes OR tools cannot find a feasible solution?
        # By letting it fail we do not get total results, but we dcan retry by the caching mechanism
        print("Exception occured")
        print(e)
        return None
def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, epoch=None, is_local=True, mtl=False):
    from nets.attention_model import AttentionModel
    
    from nets.pointer_network import PointerNetwork

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)
   


    args = load_args(os.path.join(path, 'args.json'))

    
    if is_local:
        if mtl:
            from nets.attention_mtl import AttentionModel
            model= AttentionModel(
            args['embedding_dim'],
            args['hidden_dim'],
            load_problem('local'),
            n_encode_layers=args['n_encode_layers'],
            mask_inner=True,
            mask_logits=True,
            normalization=args['normalization'],
            tanh_clipping=args['tanh_clipping'],
            checkpoint_encoder=args.get('checkpoint_encoder', False),
            shrink_size=args.get('shrink_size', None),
            context_dim=args['context_dim']
        )
        else:
            from nets.attention_local import AttentionModel
            model= AttentionModel(
            args['embedding_dim'],
            args['hidden_dim'],
            load_problem('local'),
            n_encode_layers=args['n_encode_layers'],
            mask_inner=True,
            mask_logits=True,
            normalization=args['normalization'],
            tanh_clipping=args['tanh_clipping'],
            checkpoint_encoder=args.get('checkpoint_encoder', False),
            shrink_size=args.get('shrink_size', None),
        )

    else:
        raise NotImplementedError
        problem = load_problem(args['problem']) 
        model_class = {
            'attention': AttentionModel,
            'pointer': PointerNetwork
        }.get(args.get('model', 'attention'), None)
        assert model_class is not None, "Unknown model: {}".format(model_class)
    
        model = model_class(
            args['embedding_dim'],
            args['hidden_dim'],
            problem,
            n_encode_layers=args['n_encode_layers'],
            mask_inner=True,
            mask_logits=True,
            normalization=args['normalization'],
            tanh_clipping=args['tanh_clipping'],
            checkpoint_encoder=args.get('checkpoint_encoder', False),
            shrink_size=args.get('shrink_size', None)
        )
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


######## LCP-TSP ##########
def decomposition(seeds, coordinate_dim, revision_len, offset, shift_len = 1):
    # change decomposition point
    seeds = torch.cat([seeds[:, shift_len:],seeds[:, :shift_len]], 1)

    if offset!=0:
        decomposed_seeds = seeds[:, :-offset]
        offset_seeds = seeds[:,-offset:]
    else:
        decomposed_seeds = seeds
        offset_seeds = None
    # decompose original seeds
    decomposed_seeds = decomposed_seeds.reshape(-1, revision_len, coordinate_dim)
    return decomposed_seeds, offset_seeds

def revision(revision_cost_func, reviser, decomposed_seeds, original_subtour):

    # tour length of segment TSPs
    init_cost = revision_cost_func(decomposed_seeds, original_subtour)
    cost_revised, sub_tour = reviser(decomposed_seeds, return_pi=True)
    reduced_cost = init_cost - cost_revised

    # revision_len = decomposed_seeds.shape[1]
    # reduced_cost_mean = reduced_cost[reduced_cost>0].mean()
    # print(f'revisor{revision_len}; reduced_cost{reduced_cost_mean}')

    # preserve previous tour if reduced cost is negative
    # print(original_subtour)

    sub_tour[reduced_cost < 0] = original_subtour
    decomposed_seeds = decomposed_seeds.gather(1, sub_tour.unsqueeze(-1).expand_as(decomposed_seeds))

    return decomposed_seeds

def LCP_TSP(
    seeds,
    cost_func,
    reviser,
    revision_len,
    revision_iter
    ):
    
    # width, problem_size, 2 (for TSP)
    batch_size, num_nodes, coordinate_dim = seeds.shape


    offset = num_nodes % revision_len
    if torch.cuda.is_available():
        original_subtour = torch.arange(0, revision_len, dtype=torch.long).cuda()
    else:
        original_subtour = torch.arange(0, revision_len, dtype=torch.long)
    for _ in range(revision_iter):
        decomposed_seeds, offset_seed = decomposition(seeds, 
                                                    coordinate_dim,
                                                    revision_len,
                                                    offset
                                                    )
        
        decomposed_seeds_revised = revision(cost_func, reviser, decomposed_seeds, original_subtour)

        seeds = decomposed_seeds_revised.reshape(batch_size, -1, coordinate_dim)

        if offset_seed is not None:
            seeds = torch.cat([seeds,offset_seed], dim=1)
    return seeds


def reconnect( 
        get_cost_func, 
        get_cost_func2,
        batch,
        pi_batch, 
        opts, 
        revisers,
    ):
    cost, _ = get_cost_func(batch, pi_batch)

    # instance shape: (width, problem_size, 2)
    # pi shape: (width, problem_size), dtype: torch.int64
    # cost shape: (width, )
    assert opts.problem =='tsp'

    seed = batch.gather(1, pi_batch.unsqueeze(-1).expand_as(batch))
    # seed shape (width, problem_size, 2)

    # mincosts, argmincosts = cost.min(0)
    print('cost before revision:', cost.mean().item())
    
    for revision_id in range(len(revisers)):
        seed = LCP_TSP(
            seed, 
            get_cost_func2,
            revisers[revision_id],
            opts.revision_lens[revision_id],
            opts.revision_iters[revision_id]
            )
        

        cost_revised = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
        
        print(f'after revision {revision_id}', cost_revised.mean().item())

    return seed, cost, cost_revised

def sample_many():
    raise NotImplementedError





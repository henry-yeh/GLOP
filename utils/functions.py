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
import time

def load_problem(name):
    from problems import TSP, LOCAL
    problem = {
        'local': LOCAL,
        'tsp': TSP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem

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


def load_model(path, epoch=None, is_local=True):

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
    cost_revised1, sub_tour1, cost_revised2, sub_tour2 = reviser(decomposed_seeds, return_pi=True)
    cost_revised, better_tour_idx = torch.stack((cost_revised1, cost_revised2)).min(dim=0)
    sub_tour = torch.stack((sub_tour1, sub_tour2))[better_tour_idx, torch.arange(sub_tour1.shape[0])]
    reduced_cost = init_cost - cost_revised

    # reduced_cost_mean = reduced_cost[reduced_cost>0].mean()
    # print(f'revisor{revision_len}; reduced_cost{reduced_cost_mean}')

    sub_tour[reduced_cost < 0] = original_subtour
    decomposed_seeds = decomposed_seeds.gather(1, sub_tour.unsqueeze(-1).expand_as(decomposed_seeds))

    return decomposed_seeds

def LCP_TSP(
    seeds,
    cost_func,
    reviser,
    revision_len,
    revision_iter,
    opts,
    shift_len
    ):
    
    # width, problem_size, 2 (for TSP)
    batch_size, num_nodes, coordinate_dim = seeds.shape


    offset = num_nodes % revision_len

    for _ in range(revision_iter):

        decomposed_seeds, offset_seed = decomposition(seeds, 
                                        coordinate_dim,
                                        revision_len,
                                        offset,
                                        shift_len
                                        )
        if torch.cuda.is_available():
            original_subtour = torch.arange(0, revision_len, dtype=torch.long).cuda()
        else:
            original_subtour = torch.arange(0, revision_len, dtype=torch.long)
    
        decomposed_seeds_revised = revision(cost_func, reviser, decomposed_seeds, original_subtour)
        

        seeds = decomposed_seeds_revised.reshape(batch_size, -1, coordinate_dim)

        if offset_seed is not None:
            seeds = torch.cat([seeds,offset_seed], dim=1)
    return seeds


def reconnect( 
        get_cost_func,
        batch,
        opts, 
        revisers,
    ):

    seed = batch

    for revision_id in range(len(revisers)):
        start_time = time.time()

        seed = LCP_TSP(
            seed, 
            get_cost_func,
            revisers[revision_id],
            opts.revision_lens[revision_id],
            opts.revision_iters[revision_id],
            opts=opts,
            shift_len=opts.shift_lens[revision_id]
            )
        
        cost_revised = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)

        cost_revised, cost_revised_minidx = cost_revised.reshape(-1, opts.eval_batch_size).min(0) # width, bs
        
        # cur_width = cost_revised.shape[0]

        # if revision_id < len(opts.top_k) and opts.top_k[revision_id] < cur_width:
        #     k = opts.top_k[revision_id]

        duration = time.time() - start_time

        print(f'after construction {revision_id}', cost_revised.mean().item(), f'duration {duration} \n')

    # print(seed.shape)

    # if not opts.disable_improve:
    #     seed = seed.reshape(-1, opts.eval_batch_size, opts.problem_size, 2)
    
    #     # seed shape (width, bs, ps, 2)
    #     seed = seed[cost_revised_minidx, torch.arange(opts.eval_batch_size)]

    #     start_time = time.time()
    #     for revision_id in range(len(revisers2)):
    #         seed = LCP_TSP(
    #             seed, 
    #             get_cost_func2,
    #             revisers2[revision_id],
    #             opts.revision_lens2[revision_id],
    #             opts.revision_iters2[revision_id],
    #             mood='improve',
    #             opts = opts,
    #             shift_len=opts.shift_lens2[revision_id]
    #             )
        
    #         duration = time.time() - start_time
    #         cost_revised = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)

    #         cost_revised, _ = cost_revised.reshape(-1, opts.eval_batch_size).min(0)
    #         print(f'after improvement {revision_id}', cost_revised.mean().item(), f'duration {duration} \n')
    return seed, cost_revised

def sample_many():
    raise NotImplementedError





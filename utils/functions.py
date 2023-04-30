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

    assert opts.cpus is not None
    num_cpus = opts.cpus

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

def coordinate_transformation(x):
    input = x.clone()
    max_x, indices_max_x = input[:,:,0].max(dim=1)
    max_y, indices_max_y = input[:,:,1].max(dim=1)
    min_x, indices_min_x = input[:,:,0].min(dim=1)
    min_y, indices_min_y = input[:,:,1].min(dim=1)
    # shapes: (batch_size, ); (batch_size, )
    
    diff_x = max_x - min_x
    diff_y = max_y - min_y
    xy_exchanged = diff_y > diff_x

    # shift to zero
    input[:, :, 0] -= (min_x).unsqueeze(-1)
    input[:, :, 1] -= (min_y).unsqueeze(-1)

    # exchange coordinates for those diff_y > diff_x
    input[xy_exchanged, :, 0], input[xy_exchanged, :, 1] =  input[xy_exchanged, :, 1], input[xy_exchanged, :, 0]
    
    # scale to (0, 1)
    scale_degree = torch.max(diff_x, diff_y)
    scale_degree = scale_degree.view(input.shape[0], 1, 1)
    input /= scale_degree + 1e-10
    return input

def revision(opts, revision_cost_func, reviser, decomposed_seeds, original_subtour, iter=None, embeddings=None):

    # tour length of segment TSPs
    reviser_size = original_subtour.shape[0]
    init_cost = revision_cost_func(decomposed_seeds, original_subtour)
    
    # coordinate transformation
    transformed_seeds = coordinate_transformation(decomposed_seeds)
    # augmentation
    if not opts.no_aug:
        seed2 = torch.cat((1 - transformed_seeds[:, :, [0]], transformed_seeds[:, :, [1]]), dim=2)
        seed3 = torch.cat((transformed_seeds[:, :, [0]], 1 - transformed_seeds[:, :, [1]]), dim=2)
        seed4 = torch.cat((1 - transformed_seeds[:, :, [0]], 1 - transformed_seeds[:, :, [1]]), dim=2)
        augmented_seeds = torch.cat((transformed_seeds, seed2, seed3, seed4), dim=0)
    else:
        augmented_seeds = transformed_seeds

    if iter is None:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2 = reviser(augmented_seeds, return_pi=True)
    elif iter == 0:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2, embeddings = reviser(augmented_seeds, return_pi=True, return_embedding=True)
    else:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2 = reviser(augmented_seeds, return_pi=True, embeddings=embeddings)
    
    if not opts.no_aug:
        _, better_tour_idx = torch.cat([cost_revised1, cost_revised2], dim=0).reshape(8,-1).min(dim=0)
        sub_tour = torch.cat([sub_tour1, sub_tour2], dim=0).reshape(8,-1, reviser_size)[better_tour_idx, torch.arange(sub_tour1.shape[0]//4), :]
    else:
        _, better_tour_idx = torch.stack((cost_revised1, cost_revised2)).min(dim=0)
        sub_tour = torch.stack((sub_tour1, sub_tour2))[better_tour_idx, torch.arange(sub_tour1.shape[0])]

    cost_revised, _ = reviser.problem.get_costs(decomposed_seeds, sub_tour)
    reduced_cost = init_cost - cost_revised
    
    sub_tour[reduced_cost < 0] = original_subtour
    decomposed_seeds = decomposed_seeds.gather(1, sub_tour.unsqueeze(-1).expand_as(decomposed_seeds))
    
    if embeddings is not None:
        if not opts.no_aug:
            embeddings = embeddings.gather(1, sub_tour.repeat(4, 1).unsqueeze(-1).expand_as(embeddings))
        else:
            embeddings = embeddings.gather(1, sub_tour.unsqueeze(-1).expand_as(embeddings))

    return decomposed_seeds, embeddings

def LCP_TSP(
    seeds,
    cost_func,
    reviser,
    revision_len,
    revision_iter,
    opts,
    shift_len
    ):
    
    batch_size, num_nodes, coordinate_dim = seeds.shape
    offset = num_nodes % revision_len
    embeddings = None # used only in case problem_size == revision_len for efficiency
    for i in range(revision_iter):

        decomposed_seeds, offset_seed = decomposition(seeds, 
                                        coordinate_dim,
                                        revision_len,
                                        offset,
                                        shift_len
                                        )

        original_subtour = torch.arange(0, revision_len, dtype=torch.long).to(decomposed_seeds.device)

        if revision_len == num_nodes:
            decomposed_seeds_revised, embeddings = revision(opts, cost_func, reviser, decomposed_seeds, original_subtour, iter=i, embeddings=embeddings)
            embeddings = torch.cat([embeddings[:, shift_len:],embeddings[:, :shift_len]], 1) # roll the embeddings
        else:
            decomposed_seeds_revised, _ = revision(opts, cost_func, reviser, decomposed_seeds, original_subtour)

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
    problem_size = seed.size(1) 
    if len(revisers) == 0:
        cost_revised = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
    
    for revision_id in range(len(revisers)):
        assert opts.revision_lens[revision_id] <= seed.size(1)
        start_time = time.time()
        shift_len = max(opts.revision_lens[revision_id]//opts.revision_iters[revision_id], 1)
        seed = LCP_TSP(
            seed, 
            get_cost_func,
            revisers[revision_id],
            opts.revision_lens[revision_id],
            opts.revision_iters[revision_id],
            opts=opts,
            shift_len=shift_len
            )
        cost_revised = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)      
        duration = time.time() - start_time
        
        if revision_id == 0 and not opts.no_prune: # eliminate the underperforming ones after the first round of revisions
            cost_revised, cost_revised_minidx = cost_revised.reshape(-1, opts.eval_batch_size).min(0) # width, bs
            seed = seed.reshape(-1, opts.eval_batch_size, seed.shape[-2], 2)[cost_revised_minidx, torch.arange(opts.eval_batch_size)]
    if opts.no_prune:
            cost_revised, cost_revised_minidx = cost_revised.reshape(-1, opts.eval_batch_size).min(0)
            seed = seed.reshape(-1, opts.eval_batch_size, seed.shape[-2], 2)[cost_revised_minidx, torch.arange(opts.eval_batch_size)]
    assert cost_revised.shape == (opts.eval_batch_size,)
    assert seed.shape == (opts.eval_batch_size, problem_size, 2)
        
    return seed, cost_revised


def sample_many():
    raise NotImplementedError





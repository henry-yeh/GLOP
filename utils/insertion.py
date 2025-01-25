import torch
import numpy as np
import random_insertion as ri

def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    if isinstance(arr, list):
        return np.array(arr)
    else:
        return arr

def random_insertion(cities, order=None):
    cities = _to_numpy(cities)
    order = _to_numpy(order)
    return ri.tsp_random_insertion(cities, order)

def random_insertion_parallel(cities, orders):
    cities = _to_numpy(cities)
    orders = _to_numpy(orders)
    return ri.tsp_random_insertion_parallel(cities, orders)

def random_insertion_non_euclidean(distmap, order):
    distmap = _to_numpy(distmap)
    order = _to_numpy(order)
    return ri.atsp_random_insertion(distmap, order)
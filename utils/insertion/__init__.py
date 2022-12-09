from typing import Union
from . import insertion
import numpy as np
import torch


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    else:
        return arr


def random_insertion(cities, order):

    assert len(cities.shape) == 2 and cities.shape[1] == 2
    citycount = cities.shape[0]
    if order is None:
        order = np.arange(citycount, dtype=np.uint32)
    else:
        assert len(order.shape) == 1 and order.shape[0] == cities.shape[0]
        order = _to_numpy(order).astype(np.uint32)

    if cities.dtype is not np.float32:
        cities = _to_numpy(cities).astype(np.float32)

    result, cost = insertion.random(cities, order)

    return result, cost

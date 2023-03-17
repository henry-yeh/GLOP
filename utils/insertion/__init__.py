from typing import Union
try:
    from . import insertion
except:
    import insertion
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

def cvrp_random_insertion(customerpos, depotpos, demands, capacity, order = None, exploration = 0.7):
    assert len(customerpos.shape) == 2 and customerpos.shape[1] == 2
    assert isinstance(capacity, int)

    ccount = customerpos.shape[0]
    if order is None:
        order = np.arange(ccount, dtype=np.uint32)
    else:
        assert len(order.shape) == 1 and order.shape[0] == ccount
        order = _to_numpy(order).astype(np.uint32)
    
    if isinstance(depotpos, tuple):
        assert len(depotpos)==2
        depotx, depoty = depotpos
    else:
        assert len(depotpos.shape)==1 and depotpos.shape[0]==2
        depotx, depoty = depotpos[0].item(), depotpos[1].item()
    depotx, depoty = float(depotx), float(depoty)

    customerpos = _to_numpy(customerpos)
    if customerpos.dtype is not np.float32:
        customerpos = customerpos.astype(np.float32)
    demands = _to_numpy(demands)
    if demands.dtype is not np.uint32:
        demands = demands.astype(np.uint32)
    order, sep = insertion.cvrp_random(customerpos, depotx, depoty, demands, capacity, order, exploration)
    routes = [order[i:j] for i,j in zip(sep, sep[1:])]
    return routes

def cvrplib_random_insertion(positions, demands, capacity, order = None, exploration = 0.7):
    customerpos = positions[1:]
    depotpos = positions[0]
    demands = demands[1:]
    order = np.delete(order, order==0) - 1
    routes = cvrp_random_insertion(customerpos, depotpos, demands, capacity, order, exploration)
    for r in routes:
        r += 1
    return routes


if __name__=="__main__":
    n = 51
    pos = np.random.rand(n, 2)
    depotpos = pos.mean(axis=0)
    demands = np.random.randint(1, 7, size = n)
    capacity = 30
    order=np.arange(0, n, dtype=np.uint32)
    print(*cvrplib_random_insertion(pos, demands, capacity, order), sep='\n')
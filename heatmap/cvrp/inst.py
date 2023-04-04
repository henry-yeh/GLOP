import torch
from torch_geometric.data import Data
import numpy as np

def gen_inst(n, device):
    CAPACITIES = {
        1000: 200.,
        2000: 300.,
        5000: 300.,
        7000: 300.
    }
    coors = torch.rand(size=(n+1, 2), device=device)
    demand = torch.randint(1, 10, (n+1,), device=device)
    demand[0] = 0
    capacity = CAPACITIES[n]
    return coors, demand, capacity

def gen_distance_matrix(coordinates):
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    return distances

def gen_cos_sim_matrix(shift_coors):
    dot_products = torch.mm(shift_coors, shift_coors.t())
    magnitudes = torch.sqrt(torch.sum(shift_coors ** 2, dim=1)).unsqueeze(1)
    magnitude_matrix = torch.mm(magnitudes, magnitudes.t()) + 1e-10
    cosine_similarity_matrix = dot_products / magnitude_matrix
    return cosine_similarity_matrix

def gen_pyg_data(coors, demand, capacity, k_sparse):
    n_nodes = demand.size(0)
    norm_demand = demand / capacity
    shift_coors = coors - coors[0]
    _x, _y = shift_coors[:, 0], shift_coors[:, 1]
    r = torch.sqrt(_x**2 + _y**2)
    theta = torch.atan2(_y, _x)
    x = torch.stack((norm_demand, r, theta)).transpose(1, 0)
    
    euc_mat = gen_distance_matrix(coors)
    cos_mat = gen_cos_sim_matrix(shift_coors)
    topk_values, topk_indices = torch.topk(cos_mat, 
                                           k=k_sparse, 
                                           dim=1, largest=True)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                repeats=k_sparse),
        torch.flatten(topk_indices)
        ])
    edge_attr1 = topk_values.reshape(-1, 1)
    edge_attr2 = cos_mat[edge_index[0], edge_index[1]].reshape(k_sparse*n_nodes, 1)
    edge_attr = torch.cat((edge_attr1, edge_attr2), dim=1)
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data

def trans_tsp(coors, routes, min_reviser_size=20):
    tsp_pis = []
    n_tsps_per_route = []
    for route in routes:
        start = 0
        sub_route_count = 0
        for idx, node in enumerate(route):
            if idx == 0:
                continue
            if node == 0:
                if route[idx-1] != 0:
                    tsp_pis.append(route[start: idx])
                    sub_route_count += 1
                start = idx
        n_tsps_per_route.append(sub_route_count)
    max_tsp_len = max([len(tsp_pis[i]) for i in range(len(tsp_pis))])
    max_tsp_len = max(min_reviser_size, max_tsp_len)
    padded_tsp_pis = []
    for pi in tsp_pis:
        padded_pi = torch.nn.functional.pad(pi, (0, max_tsp_len-len(pi)), mode='constant', value=0)
        padded_tsp_pis.append(padded_pi)
    padded_tsp_pis = torch.stack(padded_tsp_pis)
    tsp_insts = coors[padded_tsp_pis]
    assert tsp_insts.shape == (sum(n_tsps_per_route),max_tsp_len, 2)
    return tsp_insts, n_tsps_per_route

def sum_cost(costs, n_tsps_per_route):
    assert len(costs) == sum(n_tsps_per_route)
    if not isinstance(costs, torch.Tensor):
        costs = torch.tensor(costs)
    ret = []
    start = 0
    for n in n_tsps_per_route:
        ret.append(costs[start: start+n].sum())
        start += n
    return torch.stack(ret)
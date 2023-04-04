import torch
from torch_geometric.data import Data

K_n = {
    20: 2,
    50: 3,
    100: 4,
    500: 9,
    1000: 12,
    5000: 20,
    10000: 38
}

def gen_prizes(n, device):
    prizes = torch.rand(size=(n,), device=device) * 4 / n
    return torch.cat((torch.tensor([0.], device=device), prizes))

def gen_penalties(n, device):
    K = K_n[n]
    beta = torch.rand(size=(n,), device=device) * 3 * K / n
    return torch.cat((torch.tensor([0.], device=device), beta)) # (n+1,)

def gen_distance_matrix(coordinates):
    n_nodes = len(coordinates)
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    return distances

def gen_inst(n, device):
    coor = torch.rand((n+1, 2), device=device)
    dist_mat = gen_distance_matrix(coor)
    prizes = gen_prizes(n, device)
    penalties = gen_penalties(n, device)
    return coor, dist_mat, prizes, penalties

def gen_pyg_data(prizes, penalties, dist_mat, k_sparse):
    n_nodes = prizes.size(0)
    topk_values, topk_indices = torch.topk(dist_mat, 
                                           k=k_sparse, 
                                           dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                repeats=k_sparse),
        torch.flatten(topk_indices)
        ])
    edge_attr = topk_values.reshape(-1, 1)
    x = torch.stack((prizes, penalties)).permute(1, 0) # (n+1, 2)
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data

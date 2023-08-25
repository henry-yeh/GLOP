import torch
from torch_geometric.data import Data
torch.set_printoptions(threshold=10000)

NC = 7
K_SPARSE = 50

def gen_distance_matrix(coordinates):
    n_nodes = len(coordinates)
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    # distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e9
    return distances

def gen_inst(n):
    coor = torch.rand((n+1, 2))
    coor[0, :] = 2 # set the dummy depot relatively far away for valid covering
    dist_mat = gen_distance_matrix(coor)
    _, cover_map = torch.topk(dist_mat, k=NC+1, dim=1, largest=False) # NC+1 to also cover self
    cover_map[0, :] = 0 # dummy depot only covers itself
    return coor, dist_mat, cover_map

def gen_pyg_data(coor, dist_mat, cover_map):
    n_nodes = dist_mat.size(0)
    
    topk_values, topk_indices = torch.topk(dist_mat, 
                                           k=K_SPARSE, 
                                           dim=1, largest=False)
    edge_index = torch.stack([
        torch.repeat_interleave(torch.arange(n_nodes).to(topk_indices.device),
                                repeats=K_SPARSE),
        torch.flatten(topk_indices)
        ])
    
    edge_attr1 = topk_values.view(-1, 1)
    edge_attr2 = torch.ones_like(topk_values)
    attr2 = (torch.tensor([[i / (NC+1) for i in range(NC+1)]])).repeat(n_nodes - 1, 1)
    edge_attr2[1:, :NC+1] = attr2
    edge_attr2 = edge_attr2.view(-1, 1)
    edge_attr = torch.cat((edge_attr1, edge_attr2), dim=-1)
    
    x = torch.zeros((n_nodes, 1))
    x[0] = 1
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_data


if __name__ == '__main__':
    n = 100
    coor, dist_mat, cover_map = gen_inst(n)
    pyg_data = gen_pyg_data(coor, dist_mat, cover_map)
    
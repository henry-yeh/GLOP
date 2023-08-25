from heatmap.pctsp.inst import gen_pyg_data, gen_distance_matrix
from nets.partition_net import Net
import torch

# EPS = 1e-10

# k_sparse_table = {
#     500: 50,
#     1000: 100,
#     5000: 200,
# }
# def load_partitioner(n, device):
#     net = Net(32, 2, k_sparse=k_sparse_table[n]).to(device)
#     net.load_state_dict(torch.load(f'./pretrained/Partitioner/pctsp/pctsp-{n}.pt', map_location=device))
#     return net

# @torch.no_grad()
# def infer(model, prizes, penalties, coor):
#     n = prizes.size(0)-1
#     dist_mat = gen_distance_matrix(coor)
#     pyg_data = gen_pyg_data(prizes, penalties, dist_mat, k_sparse_table[n])
#     heatmap = model.reshape(pyg_data, model(pyg_data)) + EPS
#     return heatmap
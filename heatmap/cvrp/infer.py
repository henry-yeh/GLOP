import sys
import torch
sys.path.insert(0, './')
from heatmap.cvrp.inst import gen_pyg_data
from heatmap.cvrp.train import infer_heatmap
from nets.partition_net import Net

EPS = 1e-10

K_SPARSE = {
    1000: 100,
    2000: 200,
}
def load_partitioner(n, device, ckpt_path):
    net = Net(48, 3, K_SPARSE[n], 2)
    if ckpt_path == '':
        ckpt_path = f'./pretrained/Partitioner/cvrp/cvrp-{n}.pt'
    ckpt = torch.load(ckpt_path, map_location=device)
    print('  [*] Loading model from {}'.format(ckpt_path))
    if ckpt.get('model_state_dict', None):
        net.load_state_dict(ckpt['model_state_dict'])
    else:
        net.load_state_dict(ckpt)
    return net.to(device)

@torch.no_grad()
def infer(model, coors, demand, capacity):
    model.eval()
    n = demand.size(0)-1
    pyg_data = gen_pyg_data(coors, demand, capacity, K_SPARSE[n])
    heatmap = infer_heatmap(model, pyg_data)
    return heatmap

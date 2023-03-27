import torch

def eval(subsets, dist_mat):
    return torch.count_nonzero(subsets, dim=1)
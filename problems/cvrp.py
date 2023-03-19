import numpy as np
import torch
import pickle

CAPACITIES = {
    10: 20.,
    20: 30.,
    50: 40.,
    100: 50.,
    1000: 200,
    2000: 300,
    5000: 300,
    7000: 300,
    100000: 1000,
    
}

def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def disturb_dummy_coor(pi_batch, seeds):
    '''
    Args:
        pi_batch: (x, p_size)
        seeds: (x, p_size, 2)
    '''
    zeroIndices = torch.nonzero(pi_batch.eq(0))
    seeds[zeroIndices[:, 0], zeroIndices[:, 1]] += 1e-6*((torch.rand((zeroIndices.size(0), 2), device=seeds.device))+0.1)
    return seeds

def preprocess_inst(inst, pos):
    '''
    Args: 
        inst: n_subtours: var_size, inst_wise_max_subtour_len)
        pos: p_size+1, 2
    return: (sum_n_subtours for all width, inst_wise_max_subtour_len, 2)
    '''
    pos = torch.tensor(pos, dtype=torch.float32)
    pi_inst = torch.tensor(inst, dtype=torch.long) # (n_subtours, max_subtour_len)
    seeds = pos[pi_inst] # (sum_n_subtours, max_subtour_len, 2)
    seeds = disturb_dummy_coor(pi_inst, seeds)
    return seeds

def check_feasibility(pi_all, inst_list):
    for i in range(len(inst_list)):
        pi = pi_all[i]
        inst = inst_list[i]
        for sb_id, subtour in enumerate(pi):
            sd = 0
            for node in subtour:
                assert 1 <= inst[1][node] <= 10
                sd += inst[1][node]
            assert sd <= CAPACITIES[len(inst_list[0][0])-1]

def init(opts, insertion, inst_list=None):
    if inst_list is None:
        inst_list = load_dataset(opts.path)[:opts.val_size]
    pi_all = [insertion(*instance, None) for instance in inst_list]  # (n_dataset, n_subtours: var_len, n_nodes: var_len)
    # check_feasibility(pi_all, inst_list)
    subtours_array_list = []
    # inst-wise padding here
    for inst_id in range(len(inst_list)):
        pi_inst = pi_all[inst_id] # a list of of var-len np.arrays
        max_seq_len = max([len(pi_inst[j]) for j in range(len(pi_inst))])
        # append to max_seq_len and transform into TSP tensors
        subtours_list = []
        for subtour in pi_inst:
            subtour = np.append(subtour.astype(np.int64), np.zeros(shape=(max_seq_len+1-len(subtour),)))
            subtours_list.append(subtour)
        subtours_array = np.stack(subtours_list)
        subtours_array_list.append(subtours_array)
    # subtours_array_list: (n_dataset, n_subtours: var_len, n_nodes: var_len)
    seeds_tensor_list = []
    for inst_id, inst in enumerate(subtours_array_list):
        inst_node_coor = inst_list[inst_id][0]
        inst = preprocess_inst(inst, inst_node_coor) # tensor (n_subtours, inst_wise_max_subtour_len, 2)
        seeds_tensor_list.append(inst)
    return seeds_tensor_list
        

if __name__ == '__main__':
    dataset = load_dataset('data/vrp/vrp1000_test_seed1234.pkl')
    print(dataset)
        
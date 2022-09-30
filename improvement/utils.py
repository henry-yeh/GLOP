import numpy as np
import itertools
import random
import torch
# import math
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

def tsp_opt(points):
    """
    Dynamic programing solution for the TSP - O(2^n*n^2)
    https://gist.github.com/mlalevic/6222750

    :param points: List of (x, y) points
    :return: Optimal solution
    """

    def length(x_coord, y_coord):
        return np.linalg.norm(np.asarray(x_coord) - np.asarray(y_coord))

    # Calculate all lengths
    all_distances = [[length(x, y) for y in points] for x in points]
    # Initial value - just distance from 0 to every other point + keep the track of edges
    A = {(frozenset([0, idx+1]), idx+1): (dist, [0, idx+1])
         for idx, dist in enumerate(all_distances[0][1:])}
    cnt = len(points)
    for m in range(2, cnt):
        B = {}
        for S in [frozenset(C) | {0}
                  for C in itertools.combinations(range(1, cnt), m)]:
            for j in S - {0}:
                # This will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
                B[(S, j)] = min([(A[(S-{j}, k)][0] + all_distances[k][j], A[(S-{j}, k)][1] + [j])
                                 for k in S if k != 0 and k != j])
        A = B
    res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])

    # dist = route_distance(res, all_distances)
    # print(res)
    # print(dist)
    return np.asarray(res[1]), res[0]


def plot_tsp_solution(positions, solution):
    """
    Plot TSP solutions in 2D

    :param np.array positions: Positions of (tour_len, 2) points
    :param list solution: tour of the TSP
    :return: 2 plots (Positions and Positions+Solution)
    """

    fig, ax = plt.subplots(2, sharex=True, sharey=True)  # Prepare 2 plots
    ax[0].set_title('Raw nodes')
    ax[1].set_title('Optimised tour')
    ax[0].scatter(positions[:, 0], positions[:, 1])  # Plot A
    ax[1].scatter(positions[:, 0], positions[:, 1])  # Plot B
    start_node = 0
    distance = 0.
    N = len(solution)
    for i in range(N-1):
        start_pos = positions[start_node]
        next_node = solution[i + 1]
        end_pos = positions[next_node]
        ax[1].annotate("",
                       xy=start_pos, xycoords='data',
                       xytext=end_pos, textcoords='data',
                       arrowprops=dict(arrowstyle="<-",
                                       connectionstyle="arc3"))
        distance += np.linalg.norm(end_pos - start_pos)
        start_node = next_node

    textstr = "N nodes: %d\nTotal length: %.3f" % (N-1, distance)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=14,
               verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()


def create_tour(tour_length, seed=12345, rand=True, nearest_neighbour=False):
    """
    Create an initial tour for the TSP

    :param int tour_length: Tour length
    :param bool rand:  Generate random tour
    :param bool nearest_neighbour: Genarate nearest neighbour tour
    :return: list with a TSP tour
    """
    assert rand != nearest_neighbour,\
        "Parameters rand and nearest_neighbour cannot have the same value"
    # np.random.seed(seed=seed)
    if rand:
        N = tour_length

        tour = random.sample(range(N), N)
        # tour = np.append([0],
                         # np.random.choice(np.arange(1, N), N-1, replace=False))
        # tour = np.append(np.random.choice(np.arange(0, N), N-1, replace=False))
        return list(tour)


def calculate_distances(positions):
    """
    Calculate a all distances between poistions

    :param np.array positions: Positions of (tour_len, 2) points
    :return: list with all distances
    """

    # def length(x, y):
    #     return np.linalg.norm(np.asarray(x) - np.asarray(y))
    # distances = [[length(x, y) for y in positions] for x in positions]

    distances = distance_matrix(positions, positions)
    return distances

def route_distance(tour, distances):
    """
    Calculate a tour distance (including 0)

    :param list tour: TSP tour
    :param list : list with all distances
    :return dist: Distance of a tour
    """
    dist = 0
    prev = tour[0]
    for node in tour[1:]:
        dist += distances[int(prev)][int(node)]
        prev = node
    return dist


def swap_2opt(tour, i, k):
    """
    Swaps two edges by reversing a section of nodes

    :param list tour: TSP tour
    :param int i: First index for the swap
    :param int j: Second index for the swap
    """
    # assert tour[0] == 0 and tour[-1] != 0
    if k <= i:
        i_a = i
        i = k
        k = i_a
    assert i >= 0 and i < (len(tour) - 1)
    assert k >= i and k < len(tour)
    new_tour = tour[0:i]
    new_tour = np.append(new_tour, np.flip(tour[i:k + 1], axis=0))
    new_tour = np.append(new_tour, tour[k+1:])
    # assert len(new_tour) == len(tour)
    new_tour = [int(i) for i in new_tour]
    return list(new_tour)


def swap_2opt_(tour, i, k, tour_distance, distances):
    """
    Swaps two edges by reversing a section of nodes

    :param list tour: TSP tour
    :param int i: First index for the swap
    :param int j: Second index for the swap
    """
    # assert tour[0] == 0 and tour[-1] != 0
    if k <= i:
        i_a = i
        i = k
        k = i_a
    assert i >= 0 and i < (len(tour) - 1)
    assert k >= i and k < len(tour)

    distance = tour_distance

    aux_tour = tour.copy()
    aux_tour.append(tour[0])
    aux_tour.insert(0, tour[-1])

    distance = distance - (distances[aux_tour[i]][aux_tour[i+1]]
                           + distances[aux_tour[k+1]][aux_tour[k+2]])

    new_tour = tour[0:i]
    new_tour = np.append(new_tour, np.flip(tour[i:k + 1], axis=0))
    new_tour = np.append(new_tour, tour[k+1:])
    # assert len(new_tour) == len(tour)
    # new_tour = [int(i) for i in new_tour]
    new_tour = new_tour.astype(int).tolist()

    aux_new_tour = new_tour.copy()
    aux_new_tour.append(new_tour[0])
    aux_new_tour.insert(0, new_tour[-1])

    distance = distance + (distances[aux_new_tour[i]][aux_new_tour[i+1]]
                           + distances[aux_new_tour[k+1]][aux_new_tour[k+2]])

    return new_tour, distance


def swap_2opt_new(tour, i, k, tour_distance, D):
    """
    Swaps two edges by reversing a section of nodes

    :param list tour: TSP tour
    :param int i: First index for the swap
    :param int j: Second index for the swap
    """
    # assert tour[0] == 0 and tour[-1] != 0
    if k <= i:
        i_a = i
        i = k
        k = i_a
    assert i >= 0 and i < (len(tour) - 1)
    assert k >= i and k < len(tour)

    distance = tour_distance
    # print("tour distance before 2 opt", distance)

    if i > 0:
        pred_i = i-1
    else:
        pred_i = len(tour)-1

    if k < len(tour)-1:
        suc_k = k+1
    else:
        suc_k = 0
    # print(pred_i)
    # print(suc_k)

    remove = D[tour[pred_i]][tour[i]] + D[tour[k]][tour[suc_k]]


    # print("remove", remove)

    distance -= remove


    new_tour = tour[0:i]
    new_tour = np.append(new_tour, np.flip(tour[i:k + 1], axis=0))
    new_tour = np.append(new_tour, tour[k+1:])
    new_tour = new_tour.astype(int).tolist()

    add = D[new_tour[pred_i]][new_tour[i]] + D[new_tour[k]][new_tour[suc_k]]
    # print("add", add)
    distance += add
    # print("tour distance after 2 opt", distance)
    return new_tour, distance




# points = np.random.random((10, 2))
# distances = calculate_distances(points)
# tour = [x for x in range(0,10)]
# tour_distance = route_distance(tour, distances)
# tour_distance
# new_tour, new_tour_distance = swap_2opt_(tour, 0, 8, tour_distance, distances)
# new_tour
# new_tour_distance
# tour_distance - new_tour_distance
# new_tour
# new_tour_distance
#
# new_tour_distance_old = route_distance(new_tour, distances)
# new_tour_distance_old


def run_2opt(tour, positions, return_indices=True, return_first=True):
    """
    Improves an existing route using 2-opt until no improvement is found

    :param list tour: TSP tour
    :param list distances: distances between points (i, j)
    :param bool return_indices: return list of indices otherwise return nodes
    :param bool return_first: return just the first 2opt move
    :param bool return_first: return just the first 2opt move
    """
    improvement = True
    best_tour = tour
    distances = calculate_distances(positions)
    best_distance = route_distance(tour, distances)
    # tours: list with tours
    tours = []
    # swap_indices: list with indices to swap
    swap_indices = []
    # swap_indices: list with nodes to swap
    swap_nodes = []

    # print("initial distance", best_distance)
    while improvement:
        improvement = False
        for i in range(1, len(best_tour) - 1):
            for k in range(i+1, len(best_tour)):
                new_tour = swap_2opt(best_tour, i, k)
                new_distance = route_distance(new_tour, distances)
                if new_distance < best_distance:
                    # print(new_distance)
                    # print(new_tour)
                    swap_indices.append([i, k])
                    swap_nodes.append(sorted([new_tour[i], new_tour[k]]))
                    tours.append(best_tour)
                    best_distance = new_distance
                    best_tour = new_tour
                    improvement = True
                    break  # improvement --> return to the top of 1st loop
            # if just the first move stop the 2nd loop and stop the while loop
            if return_first and improvement:
                improvement = False
                break
            # if improvement:
            #     break
    # print("final distance", best_distance)
    assert len(best_tour) == len(tour)

    swap_indices = np.array(swap_indices)
    swap_nodes = np.array(swap_nodes)
    best_tour = np.array(best_tour)
    tours = np.array(tours)

    if return_indices:
        return best_distance, best_tour, tours, swap_indices
    else:
        return best_distance, best_tour, tours, swap_nodes


def run_2opt_policy(positions, return_first=True):
    """
    Improves an existing route using 2-opt until no improvement is found

    :param list tour: TSP tour
    :param list distances: distances between points (i, j)
    :param bool return_indices: return list of indices otherwise return nodes
    :param bool return_first: return just the first 2opt move
    :param bool return_first: return just the first 2opt move
    """
    improvement = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    best_distance = route_distance(tour, distances)
    # tours: list with tours
    tours = []
    # swap_indices: list with indices to swap
    swap_indices = []
    # swap_indices: list with nodes to swap
    swap_nodes = []

    # print("initial distance", best_distance)
    while improvement:
        improvement = False
        for i in range(0, len(best_tour) - 1):
            for k in range(i+1, len(best_tour)):
                new_tour = swap_2opt(best_tour, i, k)
                new_distance = route_distance(new_tour, distances)
                if new_distance < best_distance:
                    # print(new_distance)
                    # print(new_tour)
                    swap_indices.append([i, k])
                    swap_nodes.append(sorted([new_tour[i], new_tour[k]]))
                    tours.append(best_tour)
                    best_distance = new_distance
                    best_tour = new_tour
                    improvement = True
                    break  # improvement
            # if just the first move stop the 2nd loop and stop the while loop
            if return_first and improvement:
                improvement = False
                break
            # if improvement:
            #     break
    # print("final distance", best_distance)
    assert len(best_tour) == len(tour)

    swap_indices = np.array(swap_indices)
    swap_nodes = np.array(swap_nodes)
    best_tour = np.array(best_tour)
    tours = np.array(tours)

    return swap_indices


def index_to_action(nof_points):
    action_dic = {}
    dic_idx = 0
    for i in range(1, nof_points-1):
        for j in range(i+1, nof_points):
            action_dic[dic_idx] = np.array([i, j])
            dic_idx += 1
    return action_dic


def batch_pair_squared_dist(x, y):
    '''
    Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
    Input: x is a bxNxd matrix y is an optional bxMxd matirx
    Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
    i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2
    '''
    x_norm = (x**2).sum(2).view(x.shape[0], x.shape[1], 1)
    y_t = y.permute(0, 2, 1).contiguous()
    y_norm = (y**2).sum(2).view(y.shape[0], 1, y.shape[1])
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    dist[dist != dist] = 0  # replace nan values with 0
    dist = torch.clamp(dist, 0.0, np.inf)
    S = torch.sum(dist, dim=2)
    S_d = torch.diag_embed(1/torch.sqrt(S), offset=0, dim1=-2, dim2=-1)
    dist = torch.bmm(torch.bmm(S_d, dist), S_d)
    return dist



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name='None'):
        self.reset()
        self.name = name

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.exp_avg = 0
        self.min = 0
        self.max = 0
        self.reset_history()

    def reset_history(self):

        self.hist = {"val": [],
                     "avg": [],
                     "sum": [],
                     "count": [],
                     "exp_avg": [],
                     "min": [],
                     "max": []}

    def update(self, val, n=1, rate=0.1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if self.count > 1:
            self.exp_avg = rate * val + (1 - rate) * self.exp_avg
            if val < self.min:
                self.min = val
            if val > self.max:
                self.max = val
        else:
            self.exp_avg = val
            self.min = val
            self.max = val

        self.hist["val"].append(self.val)
        self.hist["sum"].append(self.sum)
        self.hist["count"].append(self.count)
        self.hist["avg"].append(self.avg)
        self.hist["exp_avg"].append(self.exp_avg)
        self.hist["min"].append(self.min)
        self.hist["max"].append(self.max)

    def log(self, logger):

        if self.name not in logger.keys():
            logger[self.name] = self.hist
        else:
            for key in logger[self.name].keys():
                logger[self.name][key].extend(self.hist[key])

        self.reset_history()


def run_2opt_policy_bi(positions, return_first=True):
    """
    Improves an existing route using 2-opt until no improvement is found

    :param list tour: TSP tour
    :param list distances: distances between points (i, j)
    :param bool return_indices: return list of indices otherwise return nodes
    :param bool return_first: return just the first 2opt move
    :param bool return_first: return just the first 2opt move
    """
    improvement = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    best_distance = route_distance(tour, distances)
    # tours: list with tours
    tours = []
    # swap_indices: list with indices to swap
    swap_indices = []
    # swap_indices: list with nodes to swap
    swap_nodes = []

    # print("initial distance", best_distance)
    while improvement:
        improvement = False
        for i in range(0, len(tour) - 1):
            for k in range(i+1, len(tour)):
                new_tour = swap_2opt(tour, i, k)
                new_distance = route_distance(new_tour, distances)
                # print("new_candidate_dist", new_distance)
                # print("new_cadidate_tour", new_tour)
                if new_distance < best_distance:
                    # print("accept new dist", new_distance)
                    # print("aceept_new_tour", new_tour)
                    swap_indices.append([i, k])
                    swap_nodes.append(sorted([new_tour[i], new_tour[k]]))
                    tours.append(best_tour)
                    best_distance = new_distance
                    best_tour = new_tour
                    improvement = True
                    if return_first:
                        break  # improvement --> return to the top of 1st loop

            # if just the first move stop the 2nd loop and stop the while loop
            if return_first and improvement:
                improvement = False
                break


    # print("final distance", best_distance)
    assert len(best_tour) == len(tour)

    swap_indices = np.array(swap_indices)
    swap_nodes = np.array(swap_nodes)
    best_tour = np.array(best_tour)
    tours = np.array(tours)
    if len(swap_indices) > 0:
        return swap_indices[-1]
    else:
        return None





def heuristic_2opt_fi(positions, procnum, return_dict):
    """
    Improves an existing route using 2-opt until no improvement is found

    :param list tour: TSP tour
    :param list distances: distances between points (i, j)
    :param bool return_indices: return list of indices otherwise return nodes
    :param bool return_first: return just the first 2opt move
    :param bool return_first: return just the first 2opt move
    """
    improvement = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    distances = np.rint(distances*10000)
    distances = distances.astype(int)
    best_distance = route_distance(tour, distances)
    # tours: list with tours
    tours = []
    # swap_indices: list with indices to swap
    swap_indices = []

    # print("initial distance", best_distance)
    while improvement:
        improvement = False
        for i in range(0, len(best_tour) - 1):
            for k in range(i+1, len(best_tour)):
                new_tour = swap_2opt(best_tour, i, k)
                new_distance = route_distance(new_tour, distances)
                if new_distance < best_distance:
                    swap_indices.append([i, k])
                    tours.append(best_tour)
                    best_distance = new_distance
                    best_tour = new_tour
                    improvement = True
                    break
            if improvement:
                break
    assert len(best_tour) == len(tour)

    swap_indices = np.array(swap_indices)
    best_tour = np.array(best_tour)
    tours = np.array(tours)

    return_dict[procnum] = best_tour, best_distance/10000







def heuristic_2opt_bi(positions):
    """
    Improves an existing route using 2-opt until no improvement is found

    :param list tour: TSP tour
    :param list distances: distances between points (i, j)
    :param bool return_indices: return list of indices otherwise return nodes
    :param bool return_first: return just the first 2opt move
    :param bool return_first: return just the first 2opt move
    """
    improvement = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    distances = np.rint(distances*10000)
    distances = distances.astype(int)
    best_distance = route_distance(tour, distances)
    # tours: list with tours
    tours = []
    # swap_indices: list with indices to swap
    swap_indices = []

    # print("initial distance", best_distance)
    while improvement:
        improvement = False
        for i in range(0, len(best_tour) - 1):
            for k in range(i+1, len(best_tour)):
                new_tour = swap_2opt(tour, i, k)
                new_distance = route_distance(new_tour, distances)
                if new_distance < best_distance:
                    swap_indices.append([i, k])
                    tours.append(best_tour)
                    best_distance = new_distance
                    best_tour = new_tour
                    improvement = True
        tour = best_tour
    assert len(best_tour) == len(tour)

    swap_indices = np.array(swap_indices)
    best_tour = np.array(best_tour)
    tours = np.array(tours)

    return best_tour, best_distance/10000









def heuristic_2opt_fi_restart(positions, steps):
    """
    Improves an existing route using 2-opt until no improvement is found

    :param list tour: TSP tour
    :param list distances: distances between points (i, j)
    :param bool return_indices: return list of indices otherwise return nodes
    :param bool return_first: return just the first 2opt move
    :param bool return_first: return just the first 2opt move
    """
    improvement = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    distances = np.rint(distances*10000)
    distances = distances.astype(int)
    best_distance = route_distance(tour, distances)
    restart_distance = best_distance
    # print("initial distance", best_distance)
    for n in range(steps):
        improvement = False
        for i in range(0, len(best_tour) - 1):
            for k in range(i+1, len(best_tour)):
                new_tour = swap_2opt(tour, i, k)
                new_distance = route_distance(new_tour, distances)
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_tour = new_tour
                    improvement = True
                    tour = new_tour
                    break
            if improvement:
                break
        if improvement is False:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour

            tour = create_tour(len(tour))
            best_distance = 1e10
        if n == steps-1:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour
    assert len(best_tour) == len(tour)

    return restart_tour, restart_distance/10000




def heuristic_2opt_bi_restart(positions, steps):
    """
    Improves an existing route using 2-opt until no improvement is found

    :param list tour: TSP tour
    :param list distances: distances between points (i, j)
    :param bool return_indices: return list of indices otherwise return nodes
    :param bool return_first: return just the first 2opt move
    :param bool return_first: return just the first 2opt move
    """
    improvement = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    distances = np.rint(distances*10000)
    distances = distances.astype(int)
    best_distance = route_distance(tour, distances)
    restart_distance = best_distance
    # tours: list with tours
    tours = []
    # swap_indices: list with indices to swap
    swap_indices = []

    # print("initial distance", best_distance)
    for n in range(steps):
        improvement = False
        for i in range(0, len(best_tour) - 1):
            for k in range(i+1, len(best_tour)):
                new_tour = swap_2opt(tour, i, k)
                new_distance = route_distance(new_tour, distances)
                if new_distance < best_distance:
                    swap_indices.append([i, k])
                    tours.append(best_tour)
                    best_distance = new_distance
                    best_tour = new_tour
                    improvement = True
        tour = best_tour
        if improvement is False:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour

            tour = create_tour(len(tour))
            best_distance = 1e10
        if n == steps-1:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour

    assert len(best_tour) == len(tour)

    swap_indices = np.array(swap_indices)
    best_tour = np.array(best_tour)
    tours = np.array(tours)

    return restart_tour, restart_distance/10000

import numpy as np
from scipy.optimize import linear_sum_assignment
import math
import random
import torch
import os

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def generate_distance_matrix(grid_data):

    grid_data = sorted(grid_data, key=lambda x: x["grid_id"])
    N = len(grid_data)
    dist_matrix = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i, N):
            lat1, lon1 = grid_data[i]["center_lat"], grid_data[i]["center_lon"]
            lat2, lon2 = grid_data[j]["center_lat"], grid_data[j]["center_lon"]
            d = haversine(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d  
    return dist_matrix

def read_graph_file(graph_file):

    grid_data = []
    with open(graph_file, 'r') as f:
        header = f.readline()  
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split(',')
            data = {
                "grid_id": int(parts[0]),
                "center_lat": float(parts[1]),
                "center_lon": float(parts[2])
            }
            grid_data.append(data)
    return grid_data


def compute_resource_reallocation_cost(state1, state2, distance_matrix):

    if state1.ndim == 2 and state1.shape[1] == 2:
        state1 = state1[:, 0]
    diff = state1 - state2
    supply_indices = np.where(diff == 1)[0]   
    demand_indices = np.where(diff == -1)[0]    
    
    if supply_indices.size == 0 or demand_indices.size == 0:
        return 0, (np.array([]), np.array([]))
    
    cost_matrix = distance_matrix[np.ix_(supply_indices, demand_indices)]
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()
    
    return total_cost, (supply_indices[row_ind], demand_indices[col_ind])

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    graph_file = "graph.dat"
    grid_data = read_graph_file(graph_file)
    distance_matrix = generate_distance_matrix(grid_data)

    np.random.seed(42)
    state1 = np.random.choice([0, 1], size=225)
    state2 = np.random.choice([0, 1], size=225)
    
    total_cost, assignment = compute_resource_reallocation_cost(state1, state2, distance_matrix)
    
    print( total_cost)
    print( assignment)

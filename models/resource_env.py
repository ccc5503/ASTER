import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

class ResourceEnv:
    def __init__(self, num_nodes, total_resources, k_steps):
        
        self.N = num_nodes
        self.total_resources = total_resources
        self.resources = np.zeros(( num_nodes), dtype=np.int32)   
        self.cooldowns = np.zeros((num_nodes), dtype=np.int32)  
        self.k_steps = k_steps
        nodes = np.random.choice(num_nodes, total_resources, replace=False)
        self.resources[nodes] = 1

    def step(self, actions, distance_matrix):
        """
        actions: [N],
        distance_matrix: [N, N]
        return
        new_available: [1]
        total_cost
        """

        self.cooldowns = np.maximum(0, self.cooldowns - 1)
        

        total_cost , (src, tgt) = self.compute_resource_reallocation_cost(actions, distance_matrix)
        
        
        before_count = np.sum(self.resources)

        for s, t in zip(src, tgt):
            self.resources[s] -= 1
            self.resources[t] += 1
            self.cooldowns[t] = self.k_steps
        after_count = np.sum(self.resources)
        
        assert before_count == after_count, f"Resource mismatch: before {before_count}, after {after_count}"

        new_available = np.sum((self.resources == 1) & (self.cooldowns == 0))

        return new_available , total_cost  

    def get_state(self):
        """
        return [N, 2]: states + cooldowns
        """
        return torch.tensor(np.stack([self.resources, self.cooldowns], axis=-1), dtype=torch.float32)

    def reset(self):
        """
        reset the environment
        """
        self.resources = np.zeros(self.N, dtype=np.int32)
        self.cooldowns = np.zeros(self.N, dtype=np.int32)

        nodes = np.random.choice(self.N, self.total_resources, replace=False)
        self.resources[nodes] = 1

    def compute_resource_reallocation_cost(self, actions, distance_matrix):
        """
        actions: [N]
        distance_matrix: [N, N]
        return
        total_cost:
        src: [M] 
        tgt: [M]
        """
        supply_indices = np.where((self.resources == 1) & (self.cooldowns == 0))[0]

        actions_d = np.array(actions)
        demand_indices = np.where(actions_d == 1)[0] 

        if len(supply_indices) == 0 or len(demand_indices) == 0:
            return 0, (np.array([]), np.array([]))

        cost_matrix = distance_matrix[np.ix_(supply_indices, demand_indices)]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_ind, col_ind].sum()
        return total_cost, (supply_indices[row_ind], demand_indices[col_ind])

        
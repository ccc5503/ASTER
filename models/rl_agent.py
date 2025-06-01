import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

class DQN(nn.Module):
    def __init__(self, state_dim, num_nodes, num_actions_per_node, num_objectives=4):
        super(DQN, self).__init__()
        self.num_nodes = num_nodes
        self.num_actions_per_node = num_actions_per_node
        self.num_objectives = num_objectives
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_nodes * num_actions_per_node * num_objectives)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.num_nodes, self.num_actions_per_node, self.num_objectives)
        return x  # shape: [B, N, A, 4]


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, rewards_vec, next_state, done):
        # reward shape: [4]
        self.buffer.append((state, action, rewards_vec, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, num_nodes, num_actions_per_node, replay_capacity=10000,
                 batch_size=32, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01, target_update_freq=100,device=device):
        self.state_dim = state_dim
        self.num_nodes = num_nodes
        self.num_actions_per_node = num_actions_per_node
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.total_steps = 0
        self.main_dqn = DQN(state_dim, num_nodes, num_actions_per_node).to(device)
        self.target_dqn = DQN(state_dim, num_nodes, num_actions_per_node).to(device)
        self.target_dqn.load_state_dict(self.main_dqn.state_dict())
        self.target_dqn.eval()
        self.optimizer = optim.Adam(self.main_dqn.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.device = device
        self.lambda_start=0.0
        self.lambda_end=0.6 
        self.lambda_step=5000
        self.omega_batch = None

    def select_action(self, state, omega=None,evaluate=False):
    
        if isinstance(state, torch.Tensor):
            state = state.to(self.device)
        else:
            state = torch.FloatTensor(state).to(self.device)
        
        if state.dim() == 1:
        
            state = state.unsqueeze(0) # [1, state_dim]

        if omega is None:
            omega = self.sample_omega(batch_size=1).to(self.device)

        with torch.no_grad():
            q_values = self.main_dqn(state)
            omega_exp = omega.unsqueeze(1).unsqueeze(1)

            q_scalar = torch.sum(q_values * omega_exp, dim=-1)
            
            if evaluate or np.random.rand() > self.epsilon:
                action = q_scalar.argmax(dim=-1).squeeze(0)  #shape: [N]
            else:
                action = torch.randint(0, self.num_actions_per_node, (self.num_nodes,), device=self.device)
        
        return action.to(self.device) 
    

    def select_actions_batch(self, states, env,available_resources=None, omega=None, evaluate=False):
        """
        Select dispatch actions for each node in each batch.
        Args:
            states: [B, state_dim], the states of the nodes in the batch
            env: the ResourceEnv instance, to access .resources and .cooldowns
            available_resources: [B], the number of resources that can be dispatched in each batch
            omega: preference vector for multi-objective Q values
            evaluate: if True, act greedily; otherwise ε-greedy
        Returns:
            actions: [B, N] in {0, 1}
        """
        q_values = self.main_dqn(states.to(self.device))  # [B, N, A, 4]

        if omega is None:
            omega = self.sample_omega(states.size(0)).to(self.device)
        self.omega_batch = omega.detach().clone()

        q_scalar = torch.sum(q_values * omega.unsqueeze(1).unsqueeze(1), dim=-1)  # [B, N, A]
        B, N, A = q_scalar.shape
        actions = torch.zeros((B, N), dtype=torch.long, device=self.device)

        for b in range(B):
            node_q = q_scalar[b, :, 1]  # (action=1)
            
            cooldown = torch.tensor(env.cooldowns, device=self.device)      # shape: [N]
            resources = torch.tensor(env.resources, device=self.device)

            valid_dispatch_mask = ~((resources == 1) & (cooldown > 0))

            masked_q = node_q.clone()
            masked_q[~valid_dispatch_mask] = -1e9

            if evaluate or np.random.rand() > self.epsilon:
                #
                predicted_actions = (node_q > 0).long() 
                predicted_actions[~valid_dispatch_mask] = 0 
            else:
                # ε-greedy：
                predicted_actions = torch.bernoulli(torch.full((N,), 0.05)).long().to(self.device)
                predicted_actions[~valid_dispatch_mask] = 0
            
            num_actions = predicted_actions.sum().item()
            available_k = available_resources

            if num_actions <= available_k:
                actions[b] = predicted_actions
            else:
                valid_nodes = (predicted_actions == 1)
                valid_q = masked_q[valid_nodes]
                valid_indices = torch.arange(N, device=self.device)[valid_nodes]

                if valid_q.numel() > 0:  
                    topk_idx = torch.topk(valid_q, k=available_k).indices
                    selected_nodes = valid_indices[topk_idx]
                    actions[b, selected_nodes] = 1
                else:
                    actions[b] = 0  

            selected_indices = torch.where(actions[b] == 1)[0].cpu().numpy()
            invalid_dispatch_mask = ~valid_dispatch_mask  #  (resources == 1) & (cooldown > 0)
            invalid_indices = torch.where(invalid_dispatch_mask)[0]

        return actions #shape: [B, N]



    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state.detach(), action.detach(), reward, next_state.detach(), done)

    def update(self):
        
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # sample 
        states, actions, rewards_vec, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards_vec = rewards_vec.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        omega = self.sample_omega(batch_size=self.batch_size).to(self.device)

        # calculate q_values
        q_values = self.main_dqn(states.to(self.device))  # [B, N, A, 4]
        actions = actions.unsqueeze(-1).expand(-1, self.num_nodes, 1)  # [B, N, 1]
        q_taken = torch.gather(q_values, dim=2, index=actions.unsqueeze(-1).expand(-1, -1, -1, 4))  # [B, N, 1, 4]
        q_taken = q_taken.squeeze(2).sum(dim=1)


        # calculate target_q_values
        with torch.no_grad():
            next_q = self.target_dqn(next_states.to(self.device))  # [B, N, A, 4]
            q_mean = next_q.mean(dim=-1)  # [B, N, A] -> mean over reward components
            best_action = q_mean.argmax(dim=2, keepdim=True)  # [B, N, 1]
            next_q_taken = torch.gather(next_q, dim=2, index=best_action.unsqueeze(-1).expand(-1, -1, -1, 4))  # [B, N, 1, 4]
            next_q_taken = next_q_taken.squeeze(2).sum(dim=1)  # [B, 4]
            y_vec = rewards_vec.to(self.device) + self.gamma * next_q_taken * (1 - dones.to(self.device)) # reward + gamma * next_q_taken

        # calculate loss_A
        loss_A = F.mse_loss(q_taken, y_vec)

        # calculate loss_B
        q_scalar = torch.sum(omega * q_taken, dim=1)
        y_scalar = torch.sum(omega * y_vec, dim=1)
        loss_B = F.l1_loss(q_scalar, y_scalar)
        
        # calculate total loss
        lambda_ = 0.2
        total_loss = (1 - lambda_) * loss_A + lambda_ * loss_B

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.main_dqn.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return total_loss.item()

    def get_epsilon(self):
        return self.epsilon
    
    def sample_omega(self, batch_size):
        # omega = torch.rand(batch_size, 4)
        # omega = omega / omega.sum(dim=1, keepdim=True)
        omega = torch.tensor([[0.9, 0.05, 0.03, 0.02]])
        omega = omega.repeat(batch_size, 1).to(self.device)
        return omega
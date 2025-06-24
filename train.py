import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from models import ResourceAwareDualEncoderDecoder, DQNAgent, ResourceEnv
from utils.reward import read_graph_file, generate_distance_matrix,set_random_seeds
from utils.eval import early_stopping_check,val_metric
from trainer.load_d import parse_args, init_logger, load_training_config, prepare_data, extract_config_params
from trainer.run_epoch import run_one_epoch
import numpy as np
import sys
from utils.normalizer import RewardNormalizer
from datetime import datetime
import os

def train(run_dir):
    args = parse_args()
    logger = init_logger(run_dir)
    seed = getattr(args, "seed", 42)
    set_random_seeds(seed)
    config = load_training_config(args.config, logger)
    log_dir =run_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # load data
    _, dataloader = prepare_data(config)
    data_path,T,batch_size, num_epochs, learning_rate, gamma, epsilon_start, epsilon_decay, epsilon_min, target_update_freq, total_resources, cooldown_steps, k_steps, T_short, T_long, num_nodes, graph_file,C, D_short, D_long, C_common, out_channels, num_actions_per_node = extract_config_params(config)
    grid_data = read_graph_file(graph_file)
    distance_matrix = generate_distance_matrix(grid_data) #shape: [N, N]
    location_list = [
        [item['center_lon'], item['center_lat']]
        for item in grid_data]
    location_tensor = torch.tensor(location_list, dtype=torch.float32).to(device)
    torch.cuda.empty_cache()

    # initialize model, agent, resource environment, optimizer, normalizer
    dual_predictor = ResourceAwareDualEncoderDecoder(config).to(device)
    agent = DQNAgent(state_dim=num_nodes * (C_common + 4), num_nodes=num_nodes, num_actions_per_node=num_actions_per_node, 
                     replay_capacity=10000, batch_size=batch_size, lr=learning_rate, gamma=gamma, 
                     epsilon_start=epsilon_start, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, target_update_freq=target_update_freq,device=device)
    resource_env = ResourceEnv(num_nodes, total_resources, k_steps)
    predictor_optimizer = optim.Adam(dual_predictor.parameters(), lr=learning_rate)
    reward_normalizer = RewardNormalizer(dim=4, device=device)

    # epoch level results
    episode_rewards = []
    predictor_losses = []
    reward_components_history = {
    'success': [],
    'false_alarm': [],
    'distance_cost': [],
    'aet': [] }

    best_sr = 0.0
    patience_counter = 0

    for episode in range(num_epochs):
        avg_reward, avg_loss,avg_components  = run_one_epoch(
            batch_size,dual_predictor, agent, resource_env, dataloader,
            predictor_optimizer, config, distance_matrix, location_tensor, device, episode,reward_normalizer
        )
        episode_rewards.append(avg_reward)
        predictor_losses.append(avg_loss)
        for key in reward_components_history:
            reward_components_history[key].append(avg_components[key])

        current_sr , metrics= val_metric(data_path,graph_file,T,num_nodes, k_steps,total_resources,dual_predictor,agent,logger)

        stop, best_sr, patience_counter = early_stopping_check(
            current_sr, best_sr, patience_counter, 
            threshold=0.28, 
            patience=1000, 
            logger=logger,
            dual_predictor=dual_predictor,
            agent=agent
        )
        if stop:
            break

    
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"logs/train/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    log_filename = f"{run_dir}/output_log_{timestamp}.txt"
    log_file = open(log_filename, "w", encoding="utf-8")
    sys.stdout = log_file
    train(run_dir)


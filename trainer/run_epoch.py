import torch
from trainer.run_batch import run_one_batch
import numpy as np

def safe_mean(x):
    if isinstance(x, (float, int, np.floating)):
        return x
    return np.mean([
        item.detach().cpu().item() if isinstance(item, torch.Tensor) else item
        for item in x
    ])

def run_one_epoch(batch_size ,predictor, agent, env, dataloader, optimizer, config, distance_matrix,location_tensor, device, episode,reward_normalizer):
    print(f"Episode {episode+1} starts")
    env.reset()
    total_reward = 0.0
    reward_vectors = []
    total_loss = 0.0
    component_epoch_log = {
    'success': [],
    'false_alarm': [],
    'distance_cost': [],
    'aet': [],
    'total_reward': []}

    epoch_metrics = {
    'sr': [],
    'far': [],
    'ad': [],
    'aet': [],
    'rur': [],
    'cer': []
    }
    
    for batch_idx, (st_input, lt_input, target) in enumerate(dataloader):
        batch_result = run_one_batch(
            batch_size,predictor, agent, env, st_input, lt_input, target,
            optimizer, config, distance_matrix,location_tensor, device
        ,batch_idx,reward_normalizer)

        total_reward += batch_result['reward']
        total_loss += batch_result['predictor_loss']

        for metric_name in epoch_metrics:
            epoch_metrics[metric_name].extend(batch_result['metrics'][metric_name])

        
        for key in component_epoch_log:
            component_epoch_log[key].append(batch_result['reward_components'][key])

    avg_components = {
    k: np.mean([x.item() if isinstance(x, torch.Tensor) else x for x in v])
    for k, v in component_epoch_log.items()}
    return total_reward/len(dataloader) , total_loss / len(dataloader),avg_components

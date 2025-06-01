import torch
import numpy as np
import torch.nn.functional as F


def run_one_batch(
    batch_size,
    predictor,
    agent,
    env,
    st_input,
    lt_input,
    target,
    optimizer,
    config,
    distance_matrix,
    location_tensor,
    device,
    batch_idx,
    reward_normalizer,
):
    print(f"======batch {batch_idx} start!======")

    total_rewards = 0.0
    predictor_losses = []
    reward_components_log = {
        "success": [],
        "false_alarm": [],
        "distance_cost": [],
        "aet": [],
        "total_reward": [],
        "gt_event_count": [],
    }
    epoch_metrics = {"sr": [], "far": [], "ad": [], "aet": [], "rur": [], "cer": []}
    success_window = []
    gt_event_window = []
    interval_length = 168
    sr_bonus_weight = min(300, 200 + batch_idx * 0.5)
    print("sr_bonus_weight = min(300, 200 + batch_idx * 0.5)")
 
    for i in range(st_input.shape[0]):
        print(f"======Sample {i} start!=====")
        st_sample = st_input[i].to(device)  # shape: [T_short, N, C_st]
        lt_sample = lt_input[i].to(device)  # shape: [T_long, N, C_lt]

        target_sample = target[i].to(device)  # shape: [K, N, C]

        full_res_state = env.get_state().to(
            device
        )  # shape [N, 2] 

        available_mask = (full_res_state[:, 0] == 1) & (full_res_state[:, 1] == 0)
        available = int(available_mask.sum().item())
        available_tensor = torch.tensor([available], dtype=torch.float32, device=device)
        predictions, hidden, k = predictor(
            st_sample.unsqueeze(0), lt_sample.unsqueeze(0), available_tensor, env
        )
  
        rl_state = construct_rl_state(
            hidden.squeeze(0), full_res_state, 1, location_tensor
        )  # shape: [1, N*(C_common+4)]
        actions = agent.select_actions_batch(
            rl_state, env, available_resources=available
        )
        actions_sample = actions.squeeze(0)

        env.k_steps = k.item()
        new_available, total_cost = env.step(
            actions_sample.cpu().tolist(), distance_matrix
        )

        next_st_input, next_lt_input = update_inputs(
            target_sample, st_sample, lt_sample, k
        )

        new_available_tensor = torch.tensor(
            [new_available], dtype=torch.float32, device=device
        )
        next_pred, hidden_next, _ = predictor(
            next_st_input.unsqueeze(0),
            next_lt_input.unsqueeze(0),
            new_available_tensor,
            env,
        )
        next_rl_state = construct_rl_state(
            hidden_next.squeeze(0), env.get_state().to(device), 1, location_tensor
        )

        reward, components = compute_sample_reward(
            target_sample[:, :, 0:1],
            actions_sample,
            full_res_state,
            k,
            distance_matrix,
            config,
            total_cost,
            env,
        )

        reward_vec = torch.tensor(
            [
                10 * components["success"],
                -0.5 * components["false_alarm"],
                -0.1 * components["distance_cost"],
                1 * components["aet"],
            ],
            dtype=torch.float32,
            device=device,
        )
        if hasattr(agent, "omega_batch"):
            omega_i = agent.omega_batch.squeeze(0).cpu().numpy()
            reward_i = reward_vec.cpu().numpy()

        reward_normalizer.update(reward_vec.detach())
        normalized_reward_vec = reward_normalizer.normalize(reward_vec)

        for key in reward_components_log:
            reward_components_log[key].append(components[key])
            success_window.append(components["success"])
            gt_event_window.append(components["gt_event_count"])

        # bones reward
        if (i + 1) % interval_length == 0:
            sr_interval = np.sum(success_window) / (np.sum(gt_event_window) + 1e-8)
            r_sr = sr_bonus_weight * sr_interval  

            print(f"[Interval SR Bonus] SR={sr_interval:.3f}, reward={r_sr:.2f}")

            total_rewards += r_sr

            dummy_state = torch.zeros_like(rl_state.squeeze(0))
            dummy_action = torch.zeros_like(actions_sample)
            dummy_reward_vec = torch.tensor(
                [r_sr, 1 / 2 * r_sr, 0, 0], dtype=torch.float32, device=device
            )
            for _ in range(3):
                agent.store_transition(
                    dummy_state, dummy_action, dummy_reward_vec, dummy_state, False
                )

            success_window = []
            gt_event_window = []

        agent.store_transition(
            rl_state.squeeze(0),
            actions_sample,
            normalized_reward_vec,
            next_rl_state.squeeze(0),
            False,
        )
        total_rewards += reward

        agent.update()

        predictor_loss = compute_predictor_loss(
            predictions, target_sample.unsqueeze(0), k
        )
        optimizer.zero_grad()
        predictor_loss.backward()
        optimizer.step()
        predictor_losses.append(predictor_loss.item())

        # print(f" reward: {reward}, predictor loss: {predictor_loss.item()}")

    avg_reward = total_rewards / batch_size
    avg_predictor_loss = np.mean(predictor_losses)

    # print("Batch Evaluation Metrics:")
    # print(f"Average Reward              : {avg_reward:.4f}")
    # print(f"Average Predictor Loss      : {avg_predictor_loss:.4f}")

    log = reward_components_log

    success = np.array(log["success"])
    false_alarm = np.array(log["false_alarm"])
    distance_cost = np.array(log["distance_cost"])
    aet = np.array(log["aet"])
    total_reward = np.array(log["total_reward"])
    gt_event_count = np.array(log["gt_event_count"])

    eps = 1e-8
    # SR: Success Rate
    sr = np.mean(success / (gt_event_count + eps))
    # FAR: False Alarm Rate
    far = np.mean(false_alarm / (false_alarm + success + eps))
    # AD: Average Distance
    ad = np.mean(distance_cost)
    # AET: Average Execution Time
    aet_mean = np.mean(aet)
    # RUR: Resource Utilization Rate 
    rur = np.mean(actions.sum() / 60)
    # CER: Cost-Effectiveness Ratio
    cer = np.mean(total_reward / (distance_cost + eps))

    epoch_metrics["sr"].append(sr)
    epoch_metrics["far"].append(far)
    epoch_metrics["ad"].append(ad)
    epoch_metrics["aet"].append(aet_mean)
    epoch_metrics["rur"].append(rur)
    epoch_metrics["cer"].append(cer)

    return {
        "reward": avg_reward,
        "predictor_loss": avg_predictor_loss,
        "reward_components": {k: np.mean(v) for k, v in reward_components_log.items()},
        "metrics": epoch_metrics,
    }


def construct_rl_state(hidden_rep, res_state, batch_size, location_tensor):
    """
    hidden_rep: [N, C_common]
    res_state: [N, 2]
    location_tensor: [N, 2]
     [B, N*(C_common+4)]
    """
    # res_state 的两列代表资源状态和 cooldown 状态
    location = location_tensor  # 假定 location_tensor 形状为 [N, 2]
    rl_state = torch.cat(
        [hidden_rep, res_state, location], dim=-1
    )  # shape: [N, C_common+4]
    return rl_state.view(
        batch_size, -1
    )  # 当 batch_size == 1，则变为 [1, N*(C_common+4)]


def update_inputs(target, st_input, lt_input, k):
    """
    target:   [K, N, 1]
    st_input: [T_st, N, 1]
    lt_input: [T_lt, N, 1]

    return:
        st_next: [T_st, N, 1]
        lt_next: [T_lt, N, 1]
    """
    new_steps = target[:k]  # [k, N, 1]
    st_next = torch.cat([st_input[k:], new_steps], dim=0)
    lt_next = torch.cat([lt_input[k:], new_steps], dim=0)

    return st_next, lt_next


def compute_sample_reward(
    target, action, prev_state, k, dist_matrix, config, total_cost, env
):
    """
    target: [K, N, 1] -> future K_MAX steps
    action: [N]
    prev_state: [N, 2]
    k: int
    dist_matrix: [N, N]
    -> reward
    """
    target = target[:k]
    target_bin = (target > 0.5).float()  # [K, N, 1]
    gt_event_count = (target_bin.max(dim=0).values > 0.5).sum()
    target_agg = target_bin.max(dim=0).values  # [1, N, 1] -> [N, 1]
    target_agg = target_agg.squeeze(-1)  # [N]
    action = action.float()  # [N]

    has_resource = prev_state[:, 0] == 1
    cooldown = prev_state[:, 1]
    # === Success ===
    true_positive = (action > 0) & (target_agg > 0)
    prev_dispatch_success = (
        (action == 0) & (has_resource) & (cooldown >= k) & (target_agg > 0)
    )
    success_count = true_positive.sum() + prev_dispatch_success.sum()

    # === False Alarm ===
    false_positive = (action > 0) & (target_agg == 0)
    false_alarm_count = false_positive.sum()

    # === Dispatch Distance Cost ===
    reallocation_cost = total_cost

    # === Early Time (AET)
    aet_per_node = torch.full((target_bin.shape[1],), float(k), device=target.device)
    for t in range(k):
        newly_earlier = (target_bin[t].squeeze(-1) > 0) & (aet_per_node > t)
        aet_per_node[newly_earlier] = torch.tensor(float(t), device=target.device)
    aet_vector = aet_per_node[true_positive]

    avg_aet = aet_vector.mean()
    avg_aet = (
        aet_vector.mean()
        if aet_vector.numel() > 0
        else torch.tensor(0.0, device=target.device)
    )
    # === Normalized reward components ===
    alpha = config.get("reward_alpha", 1.0)  # 
    beta = config.get("reward_beta", 0.01)  # 
    gamma = config.get("reward_gamma", 0.01)  # 
    delta = config.get("reward_delta", 0.3)  # 

    reward = (
        alpha * success_count.item()
        - beta * false_alarm_count.item()
        - gamma * reallocation_cost
        + delta * avg_aet.item()
    )
    reward_components = {
        "success": success_count.item(),
        "false_alarm": false_alarm_count.item(),
        "distance_cost": reallocation_cost,
        "aet": avg_aet.item(),
        "total_reward": reward,
        "gt_event_count": gt_event_count.item(),
    }

    return reward, reward_components


def compute_predictor_loss(predictions, target, k):
    """
    predictions: [B, K, N, 1]
    target: [B, K, N, 1]
    k: [B] — actual steps
    """
    if isinstance(predictions, list):
        predictions = torch.stack(predictions, dim=1)  # [B, K, N, 1]
        # === 1.  mask: [B, K]， k[b] True/ False

    B, K, N, _ = predictions.shape
    step_range = torch.arange(K, device=predictions.device).unsqueeze(0)  # [1, K]
    k_expand = k.unsqueeze(1)  # [B, 1]
    mask = step_range < k_expand  # [B, K] bool

    mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
    mask = mask.expand(-1, -1, N, 1)  # [B, K, N, 1]

    element_loss = F.mse_loss(
        predictions, target[:, :, :, 0:1], reduction="none"
    )  # [B, K, N, 1]

    masked_loss = element_loss * mask  # [B, K, N, 1]

    total_loss = masked_loss.sum()
    valid_count = mask.sum()

    if valid_count == 0:
        return torch.tensor(0.0, device=predictions.device)
    else:
        return total_loss / valid_count

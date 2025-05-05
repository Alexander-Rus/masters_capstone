import torch
import torch.nn.functional as F
import numpy as np

def quantile_huber_loss(predictions, targets, taus):
    """
    predictions: (batch_size, num_taus, 1)
    targets: (batch_size, num_taus, 1)
    taus: (batch_size, num_taus, 1)
    """
    td_errors = targets - predictions  # (B, N, 1)
    huber_loss = F.smooth_l1_loss(predictions, targets, reduction="none")  # (B, N, 1)
    
    tau = taus.unsqueeze(2)  # (B, N, 1, 1)
    error = td_errors.unsqueeze(1)  # (B, 1, N, 1)
    loss = (torch.abs(tau - (error.detach() < 0).float()) * huber_loss.unsqueeze(1)).mean()
    return loss

def compute_target(online_net, target_net, next_state_img, next_state_float, rewards, dones, gamma, num_taus):
    batch_size = rewards.shape[0]

    with torch.no_grad():
        taus = torch.rand(batch_size, num_taus, device=rewards.device).unsqueeze(-1)
        q_next, _ = target_net(next_state_img, next_state_float, num_taus, taus)
        q_values = q_next.mean(dim=1)
        next_actions = q_values.argmax(dim=1, keepdim=True)

        q_target, tau_target = target_net(next_state_img, next_state_float, num_taus, taus)
        q_target = q_target.gather(2, next_actions.unsqueeze(1).expand(-1, num_taus, -1))

        target = rewards.unsqueeze(1).unsqueeze(-1) + gamma * (1.0 - dones.unsqueeze(1).unsqueeze(-1)) * q_target


    return target, taus

def compute_iqn_loss(model, target_model, batch, cfg, device):
    # Unpack batch
    state_img, state_float, action, reward, next_state_img, next_state_float, done = zip(*batch)

    # Convert to tensors
    state_img = torch.tensor(np.array(state_img), dtype=torch.float32, device=device) / 255.0
    state_float = torch.tensor(np.array(state_float), dtype=torch.float32, device=device)
    action = torch.tensor(action, dtype=torch.long, device=device).unsqueeze(-1)
    reward = torch.tensor(reward, dtype=torch.float32, device=device)
    next_state_img = torch.tensor(np.array(next_state_img), dtype=torch.float32, device=device) / 255.0
    next_state_float = torch.tensor(np.array(next_state_float), dtype=torch.float32, device=device)
    done = torch.tensor(done, dtype=torch.float32, device=device)

    # Compute target
    target, taus = compute_target(model, target_model, next_state_img, next_state_float, reward, done, cfg.gamma, cfg.n_quantiles)

    # Sample taus for online network
    batch_size = reward.shape[0]
    taus_pred = torch.rand(batch_size, cfg.n_quantiles, device=device).unsqueeze(-1)

    q_pred, _ = model(state_img, state_float, cfg.n_quantiles, taus_pred)
    q_pred = q_pred.gather(2, action.unsqueeze(1).expand(-1, cfg.n_quantiles, -1))

    loss = quantile_huber_loss(q_pred, target, taus_pred)
    return loss

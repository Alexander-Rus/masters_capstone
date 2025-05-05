import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import config


class IQN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n_actions = cfg.n_actions

        # === Conv encoder ===
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # output: (32, ?)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # === Dynamically infer image output dim ===
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, cfg.frame_height, cfg.frame_width)
            dummy_output = self.conv(dummy_input)
            self.img_output_dim = dummy_output.view(1, -1).shape[1]

        # === State encoder ===
        self.fc_state = nn.Sequential(
            nn.Linear(cfg.state_float_dim, 128),
            nn.ReLU(),
        )

        # === Quantile embedding ===
        self.quantile_fc = nn.Linear(cfg.iqn_embed_dim, self.img_output_dim + 128)
        self.final_input_dim = self.img_output_dim + 128

        # === Final head ===
        self.fc_final = nn.Sequential(
            nn.Linear(self.final_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, img, state, num_quantiles, tau=None):
        batch_size = img.shape[0]

        # Encode image and state
        img_feat = self.conv(img).view(batch_size, -1)
        state_feat = self.fc_state(state)
        full_feat = torch.cat([img_feat, state_feat], dim=1)  # (B, D)

        # Sample taus
        if tau is None:
            tau = torch.rand(batch_size, num_quantiles, 1, device=img.device)

        quantile_basis = torch.arange(1, self.cfg.iqn_embed_dim + 1, device=img.device).float()
        tau_embed = tau * np.pi * quantile_basis  # (B, N, D)
        tau_feat = torch.cos(tau_embed)  # (B, N, D)
        quantile_feat = F.relu(self.quantile_fc(tau_feat))  # (B, N, final_input_dim)

        # Tile input features for each quantile
        full_feat = full_feat.unsqueeze(1).expand(-1, num_quantiles, -1)  # (B, N, D)
        full_feat = full_feat.contiguous().view(batch_size * num_quantiles, -1)
        quantile_feat = quantile_feat.view(batch_size * num_quantiles, -1)

        x = full_feat * quantile_feat
        q_values = self.fc_final(x).view(batch_size, num_quantiles, -1)

        return q_values, tau

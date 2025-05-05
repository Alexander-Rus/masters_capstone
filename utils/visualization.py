# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict

from config import config


def plot_q_values_over_time(rollout_results: Dict, inferer, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    tau = torch.linspace(0.05, 0.95, config.iqn_k)[:, None].to("cuda")

    for idx in [0, 5, 10, 20, -20]:
        if idx >= len(rollout_results["frames"]):
            continue

        q_values = []
        actions = []

        for t in range(config.temporal_mini_race_duration_actions):
            rollout_results["state_float"][idx][0] = t
            q_dist = inferer.infer_network(
                rollout_results["frames"][idx],
                rollout_results["state_float"][idx],
                tau
            )  # (k, num_actions)
            q_values.append(q_dist.mean(dim=0).cpu().numpy())
            actions.append(np.argmax(q_values[-1]))

        q_values = np.array(q_values)
        plt.figure()
        for i in range(q_values.shape[1]):
            plt.plot(q_values[:, i], label=f"Action {i}")
        plt.title(f"Q-values at frame {idx}")
        plt.xlabel("Time step")
        plt.ylabel("Q-value")
        plt.legend()
        plt.savefig(save_path / f"q_values_frame{idx}.png")
        plt.close()


def plot_quantile_distribution(rollout_results: Dict, inferer, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    tau = torch.linspace(0.05, 0.95, config.iqn_k)[:, None].to("cuda")

    idx = min(100, len(rollout_results["frames"]) - 1)
    q_dist = inferer.infer_network(
        rollout_results["frames"][idx],
        rollout_results["state_float"][idx],
        tau
    )

    action = q_dist.mean(dim=0).argmax().item()
    values = q_dist[:, action].cpu().detach().numpy()
    values.sort()

    plt.figure()
    plt.plot(tau.cpu().numpy(), values, marker='o')
    plt.title(f"Quantile Distribution for Best Action (frame {idx})")
    plt.xlabel("τ (Quantile)")
    plt.ylabel("Q(τ)")
    plt.grid(True)
    plt.savefig(save_path / "quantile_distribution.png")
    plt.close()

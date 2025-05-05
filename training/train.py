import os
import torch
import torch.optim as optim
from pathlib import Path
from collections import defaultdict

from config import config
from model.iqn import IQN
from training.loss import compute_iqn_loss
from buffer.replay_buffer import ReplayBuffer, Experience
from env.game_manager import GameManager


def train(config, env, buffer, buffer_test, online_network, target_network, optimizer, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IQN(config).to(device)
    target_model = IQN(config).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    buffer = ReplayBuffer(config.buffer_size)
    game = GameManager()

    steps = 0
    while steps < config.train_steps:
        rollout = game.rollout(model)
        buffer.add_rollout(rollout)

        for _ in range(config.updates_per_rollout):
            if len(buffer) < config.batch_size:
                continue

            batch = buffer.sample(config.batch_size)
            loss = compute_iqn_loss(model, target_model, batch, config, device)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            steps += 1

        if steps % config.target_update_interval == 0:
            target_model.load_state_dict(model.state_dict())

        if steps % config.checkpoint_freq == 0:
            ckpt_path = Path(config.save_dir) / f"checkpoint_{steps}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    train()

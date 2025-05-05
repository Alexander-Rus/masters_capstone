import torch
import numpy as np
from pathlib import Path

from config import config
from env.game_manager import GameManager
from model.iqn import IQN
from buffer.replay_buffer import make_buffers
from training.train import train


def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create buffers
    buffer, buffer_test = make_buffers(config.buffer_size)

    # Initialize networks
    online_network = IQN(config).to(device)
    target_network = IQN(config).to(device)
    target_network.load_state_dict(online_network.state_dict())
    
    optimizer = torch.optim.Adam(online_network.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    # Game environment
    env = GameManager()

    # Training loop
    train(
        config=config,
        env=env,
        buffer=buffer,
        buffer_test=buffer_test,
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer,
        scaler=scaler,
    )


if __name__ == "__main__":
    main()

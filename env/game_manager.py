import torch
import numpy as np
from config import config

class DummyTMInterface:
    def __init__(self):
        self.t = 0

    def reset(self):
        self.t = 0
        return self.get_observation()

    def step(self, action_idx):
        self.t += config.run_steps_per_action
        done = self.t >= config.max_rollout_time_ms
        return self.get_observation(), self.get_reward(), done

    def get_observation(self):
        frame = np.random.randint(0, 255, (1, config.frame_height, config.frame_width), dtype=np.uint8)
        state_float = np.random.rand(config.state_float_dim).astype(np.float32)
        return frame, state_float

    def get_reward(self):
        return np.random.rand() * 0.1


class GameManager:
    def __init__(self):
        self.env = DummyTMInterface()

    def rollout(self, model):
        device = config.device  # Ensure we use same device as model

        frames, floats, actions, rewards, q_values = [], [], [], [], []
        frame, state_float = self.env.reset()
        done = False

        while not done:
            frame_tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
            float_tensor = torch.tensor(state_float, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                q_vals, _ = model(frame_tensor, float_tensor, config.n_quantiles)
                q_val = q_vals.mean(dim=1)
                action_idx = q_val.argmax(dim=1).item()

            next_obs, reward, done = self.env.step(action_idx)

            frames.append(frame)
            floats.append(state_float)
            actions.append(action_idx)
            rewards.append(reward)
            q_values.append(q_vals.squeeze(0).cpu().numpy())

            frame, state_float = next_obs

        return {
            "frames": frames,
            "state_float": floats,
            "actions": actions,
            "rewards": rewards,
            "q_values": q_values
        }

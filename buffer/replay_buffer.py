"""
Replay Buffer for storing agent experiences and sampling for training.
"""

import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
from config import config

import numpy as np

# Define a simple experience tuple
Experience = namedtuple("Experience", [
    "state_img",          # np.ndarray (H, W)
    "state_float",        # np.ndarray (vector of floats)
    "action",             # int
    "reward",             # float
    "next_state_img",     # np.ndarray
    "next_state_float",   # np.ndarray
    "done"                # bool
])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Experience):
        self.buffer.append(experience)

    def add_rollout(self, rollout: dict):
        for i in range(len(rollout["frames"]) - 1):
            experience = Experience(
                rollout["frames"][i],
                rollout["state_float"][i],
                rollout["actions"][i],
                rollout["rewards"][i],
                rollout["frames"][i + 1],
                rollout["state_float"][i + 1],
                False  # you can update this to be True on terminal state if needed
            )
            self.add(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Optional[Experience]] = [None] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, experience: Experience):
        self.buffer[self.pos] = experience
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        if self.pos == 0 and self.buffer[-1] is None:
            valid_len = self.pos
        else:
            valid_len = self.capacity if self.buffer[-1] is not None else self.pos

        if valid_len == 0:
            return [], np.array([]), np.array([])

        probs = self.priorities[:valid_len] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(valid_len, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
            self.max_priority = max(self.max_priority, prio)

    def __len__(self):
        return sum(x is not None for x in self.buffer)

def make_buffers(capacity):
    return ReplayBuffer(capacity), ReplayBuffer(int(capacity * config.buffer_test_ratio))
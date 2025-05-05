import copy
import random
from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch
from torchrl.data import ReplayBuffer

from config_files import config_copy
from trackmania_rl import utilities


class DQN_Network(torch.nn.Module):
    def __init__(
        self,
        float_inputs_dim: int,
        float_hidden_dim: int,
        conv_head_output_dim: int,
        dense_hidden_dimension: int,
        iqn_embedding_dimension: int,  # Unused but kept for API compatibility
        n_actions: int,
        float_inputs_mean: npt.NDArray,
        float_inputs_std: npt.NDArray,
    ):
        super().__init__()
        activation = torch.nn.LeakyReLU
        self.img_head = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=4, stride=2),
            activation(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2),
            activation(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            activation(inplace=True),
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1),
            activation(inplace=True),
            torch.nn.Flatten(),
        )

        self.float_head = torch.nn.Sequential(
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            activation(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            activation(inplace=True),
        )

        input_dim = conv_head_output_dim + float_hidden_dim
        self.q_head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, dense_hidden_dimension),
            activation(inplace=True),
            torch.nn.Linear(dense_hidden_dimension, n_actions),
        )

        self.float_inputs_mean = torch.tensor(float_inputs_mean, dtype=torch.float32).to("cuda")
        self.float_inputs_std = torch.tensor(float_inputs_std, dtype=torch.float32).to("cuda")
        self.n_actions = n_actions

        self.initialize_weights()

    def initialize_weights(self):
        activation_gain = torch.nn.init.calculate_gain("leaky_relu", 1e-2)
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                utilities.init_orthogonal(m, activation_gain)

    def forward(
        self, img: torch.Tensor, float_inputs: torch.Tensor, num_quantiles: int, tau=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # The output shape is (batch_size, n_actions) repeated `num_quantiles` times for compatibility
        batch_size = img.shape[0]
        img_out = self.img_head(img)
        float_out = self.float_head((float_inputs - self.float_inputs_mean) / self.float_inputs_std)
        concat = torch.cat((img_out, float_out), dim=1)
        q_values = self.q_head(concat)  # shape: (batch_size, n_actions)

        # Repeat for compatibility with IQN interface
        q_repeated = q_values.repeat_interleave(num_quantiles, dim=0)
        tau_dummy = torch.full((batch_size * num_quantiles, 1), 0.5, device="cuda")
        return q_repeated, tau_dummy

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self


class Trainer:
    def __init__(
        self,
        online_network: DQN_Network,
        target_network: DQN_Network,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        batch_size: int,
        iqn_n: int,  # unused but accepted for API compatibility
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.scaler = scaler
        self.batch_size = batch_size

    def train_on_batch(self, buffer: ReplayBuffer, do_learn: bool) -> Tuple[float, float]:
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            batch, batch_info = buffer.sample(self.batch_size, return_info=True)
            (
                state_img,
                state_float,
                actions,
                rewards,
                next_state_img,
                next_state_float,
                gammas,
            ) = batch

            IS_weights = (
                torch.from_numpy(batch_info["_weight"]).to("cuda", non_blocking=True)
                if config_copy.prio_alpha > 0 else 1.0
            )

            q_next = self.target_network(next_state_img, next_state_float, 1)[0].max(dim=1, keepdim=True)[0]
            q_target = rewards + gammas * q_next

            q_pred = self.online_network(state_img, state_float, 1)[0].gather(1, actions)

            loss = torch.nn.functional.smooth_l1_loss(q_pred, q_target.detach(), reduction='none').squeeze()
            total_loss = (IS_weights * loss).sum() if config_copy.prio_alpha > 0 else loss.mean()

        if do_learn:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), config_copy.clip_grad_norm)
            torch.nn.utils.clip_grad_value_(self.online_network.parameters(), config_copy.clip_grad_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            grad_norm = 0.0

        return total_loss.detach().cpu().item(), grad_norm if isinstance(grad_norm, float) else grad_norm.detach().cpu().item()


class Inferer:
    def __init__(self, network, iqn_k, tau_epsilon_boltzmann):
        self.network = network
        self.iqn_k = iqn_k
        self.tau_epsilon_boltzmann = tau_epsilon_boltzmann

    def infer_network(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray, tau=None) -> npt.NDArray:
        with torch.no_grad():
            img_tensor = (torch.from_numpy(img_inputs_uint8).unsqueeze(0).to("cuda", dtype=torch.float32) - 128) / 128
            float_tensor = torch.from_numpy(float_inputs[None, :]).to("cuda")
            q_values = self.network(img_tensor, float_tensor, int(self.iqn_k))[0]
            return q_values.view(self.iqn_k, -1).cpu().numpy().astype(np.float32)

    def get_exploration_action(self, img_inputs_uint8: npt.NDArray, float_inputs: npt.NDArray) -> Tuple[int, bool, float, npt.NDArray]:
        q_values = self.infer_network(img_inputs_uint8, float_inputs).mean(axis=0)
        r = random.random()
        if self.is_explo and r < self.epsilon:
            noisy = np.random.randn(*q_values.shape)
        elif self.is_explo and r < self.epsilon + self.epsilon_boltzmann:
            noisy = q_values + self.tau_epsilon_boltzmann * np.random.randn(*q_values.shape)
        else:
            noisy = q_values

        action = np.argmax(noisy)
        greedy_action = np.argmax(q_values)
        return action, action == greedy_action, np.max(q_values), q_values


def make_untrained_dqn_network(jit: bool, is_inference: bool) -> Tuple[DQN_Network, DQN_Network]:
    net = DQN_Network(
        float_inputs_dim=config_copy.float_input_dim,
        float_hidden_dim=config_copy.float_hidden_dim,
        conv_head_output_dim=config_copy.conv_head_output_dim,
        dense_hidden_dimension=config_copy.dense_hidden_dimension,
        iqn_embedding_dimension=config_copy.iqn_embedding_dimension,
        n_actions=len(config_copy.inputs),
        float_inputs_mean=config_copy.float_inputs_mean,
        float_inputs_std=config_copy.float_inputs_std,
    )
    if jit:
        if config_copy.is_linux:
            compile_mode = None if "rocm" in torch.__version__ else ("max-autotune" if is_inference else "max-autotune-no-cudagraphs")
            compiled = torch.compile(net, dynamic=False, mode=compile_mode)
        else:
            compiled = torch.jit.script(net)
    else:
        compiled = copy.deepcopy(net)

    return compiled.to("cuda", memory_format=torch.channels_last).train(), net.to("cuda", memory_format=torch.channels_last).train()

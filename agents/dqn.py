import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        target_update_interval: int = 200,
        device: Optional[str] = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.target_update_interval = target_update_interval
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.train_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def set_epsilon_for_episode(self, episode: int, total_episodes: int) -> None:
        if total_episodes <= 1:
            self.epsilon = self.epsilon_end
            return
        progress = min(max((episode - 1) / float(total_episodes - 1), 0.0), 1.0)
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def train_step(self, replay_buffer, batch_size: int) -> float:
        if len(replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.online_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

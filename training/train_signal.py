from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer
from environment import TrafficEnv


def _save_checkpoint(agent: DQNAgent, path: Path, episode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.build_checkpoint(episode=episode), path)


def train_dqn(
    episodes: int = 2000,
    batch_size: int = 64,
    buffer_capacity: int = 50000,
    warmup_steps: int = 1000,
    render_mode: bool = False,
    checkpoint_path: Optional[str] = None,
    checkpoint_interval: int = 100,
    resume: bool = False,
) -> List[Dict[str, float]]:
    env = TrafficEnv(render_mode=render_mode)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, gamma=0.99)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    checkpoint_file = Path(checkpoint_path) if checkpoint_path else None
    start_episode = 1

    if resume and checkpoint_file is not None and checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=agent.device)
        agent.load_checkpoint(checkpoint, load_optimizer=True)
        start_episode = int(checkpoint.get("episode", 0)) + 1

    logs: List[Dict[str, float]] = []
    total_steps = 0
    reward_window: deque = deque(maxlen=100)
    waiting_window: deque = deque(maxlen=100)
    loss_window: deque = deque(maxlen=100)

    for episode in range(start_episode, episodes + 1):
        agent.set_epsilon_for_episode(episode=episode, total_episodes=episodes, decay_portion=0.9)
        state = env.reset()
        done = False
        episode_reward = 0.0
        waiting_values: List[float] = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            waiting_values.append(float(info["average_waiting_cars"]))

            if len(replay_buffer) >= max(batch_size, warmup_steps):
                loss = agent.train_step(replay_buffer, batch_size)
                if loss > 0.0:
                    loss_window.append(loss)

            total_steps += 1

        avg_waiting = float(np.mean(waiting_values)) if waiting_values else 0.0
        reward_window.append(float(episode_reward))
        waiting_window.append(avg_waiting)
        moving_avg_reward = float(np.mean(reward_window))
        moving_avg_waiting = float(np.mean(waiting_window))
        moving_avg_loss = float(np.mean(loss_window)) if loss_window else 0.0

        log_row = {
            "episode": float(episode),
            "episode_reward": float(episode_reward),
            "average_waiting_cars": avg_waiting,
            "moving_avg_reward_100": moving_avg_reward,
            "moving_avg_waiting_100": moving_avg_waiting,
            "moving_avg_loss_100": moving_avg_loss,
            "epsilon": float(agent.epsilon),
            "buffer_size": float(len(replay_buffer)),
            "total_steps": float(total_steps),
        }
        logs.append(log_row)

        if checkpoint_file is not None and checkpoint_interval > 0 and episode % checkpoint_interval == 0:
            _save_checkpoint(agent, checkpoint_file, episode)

        if episode % 50 == 0:
            print(
                f"episode={episode} reward={episode_reward:.3f} avg_waiting={avg_waiting:.3f} "
                f"ma100_reward={moving_avg_reward:.3f} ma100_waiting={moving_avg_waiting:.3f} "
                f"ma100_loss={moving_avg_loss:.4f} epsilon={agent.epsilon:.3f} buffer={len(replay_buffer)}"
            )

    if checkpoint_file is not None:
        _save_checkpoint(agent, checkpoint_file, episodes)

    env.close()
    return logs


def evaluate_dqn(
    checkpoint_path: str,
    episodes: int = 5,
    render_mode: bool = True,
) -> List[Dict[str, float]]:
    env = TrafficEnv(render_mode=render_mode)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, gamma=0.99)

    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    agent.load_checkpoint(checkpoint, load_optimizer=False)
    agent.epsilon = 0.0

    logs: List[Dict[str, float]] = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        waiting_values: List[float] = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            waiting_values.append(float(info["average_waiting_cars"]))

        avg_waiting = float(np.mean(waiting_values)) if waiting_values else 0.0
        row = {
            "episode": float(episode),
            "episode_reward": float(episode_reward),
            "average_waiting_cars": avg_waiting,
        }
        logs.append(row)
        print(
            f"eval_episode={episode} reward={episode_reward:.3f} avg_waiting={avg_waiting:.3f}"
        )

    env.close()
    return logs

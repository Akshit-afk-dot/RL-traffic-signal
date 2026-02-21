from typing import Dict, List

import numpy as np

from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer
from environment import TrafficEnv


def train_dqn(
    episodes: int = 2000,
    batch_size: int = 64,
    buffer_capacity: int = 50000,
    warmup_steps: int = 1000,
    render_mode: bool = False,
) -> List[Dict[str, float]]:
    env = TrafficEnv(render_mode=render_mode)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, gamma=0.99)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    logs: List[Dict[str, float]] = []
    total_steps = 0

    for episode in range(1, episodes + 1):
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
                agent.train_step(replay_buffer, batch_size)

            total_steps += 1

        avg_waiting = float(np.mean(waiting_values)) if waiting_values else 0.0
        log_row = {
            "episode": float(episode),
            "episode_reward": float(episode_reward),
            "average_waiting_cars": avg_waiting,
            "epsilon": float(agent.epsilon),
            "buffer_size": float(len(replay_buffer)),
            "total_steps": float(total_steps),
        }
        logs.append(log_row)

        if episode % 50 == 0:
            print(
                f"episode={episode} reward={episode_reward:.3f} avg_waiting={avg_waiting:.3f} "
                f"epsilon={agent.epsilon:.3f} buffer={len(replay_buffer)}"
            )

    env.close()
    return logs

import argparse

from training.train_signal import train_dqn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    logs = train_dqn(episodes=args.episodes, render_mode=args.render)
    if logs:
        last = logs[-1]
        print(
            "training_complete",
            f"episodes={int(last['episode'])}",
            f"final_reward={last['episode_reward']:.3f}",
            f"final_avg_waiting={last['average_waiting_cars']:.3f}",
            f"final_epsilon={last['epsilon']:.3f}",
        )


if __name__ == "__main__":
    main()

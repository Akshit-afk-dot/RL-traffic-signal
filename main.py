import argparse

from training.train_signal import evaluate_dqn, train_dqn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/traffic_signal_dqn.pt")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.mode == "train":
        logs = train_dqn(
            episodes=args.episodes,
            render_mode=args.render,
            checkpoint_path=args.checkpoint,
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
        )
        if logs:
            last = logs[-1]
            print(
                "training_complete",
                f"episodes={int(last['episode'])}",
                f"final_reward={last['episode_reward']:.3f}",
                f"final_avg_waiting={last['average_waiting_cars']:.3f}",
                f"final_epsilon={last['epsilon']:.3f}",
                f"checkpoint={args.checkpoint}",
            )
    else:
        logs = evaluate_dqn(
            checkpoint_path=args.checkpoint,
            episodes=args.episodes,
            render_mode=args.render,
        )
        if logs:
            avg_reward = sum(x["episode_reward"] for x in logs) / len(logs)
            avg_waiting = sum(x["average_waiting_cars"] for x in logs) / len(logs)
            print(
                "evaluation_complete",
                f"episodes={len(logs)}",
                f"avg_reward={avg_reward:.3f}",
                f"avg_waiting={avg_waiting:.3f}",
                f"checkpoint={args.checkpoint}",
            )


if __name__ == "__main__":
    main()

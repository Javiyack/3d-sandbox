import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC

import robot_walk  # noqa: F401 - register Gymnasium environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mostrar el sandbox biped humanoide")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = None
    if args.model is not None:
        if not args.model.exists():
            raise FileNotFoundError(f"No existe el modelo '{args.model}'")
        algorithm = args.algorithm
        if algorithm is None:
            algorithm = "sac" if "sac" in args.model.name.lower() else "ppo"
        model = (
            PPO.load(args.model, device=args.device)
            if algorithm == "ppo"
            else SAC.load(args.model, device=args.device)
        )

    env = gym.make("BipedWalk-v0", render_mode="human")
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            env.render()

        print(f"Episodio {episode + 1}: recompensa={episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()

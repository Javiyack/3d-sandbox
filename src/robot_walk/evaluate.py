import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC

import robot_walk  # noqa: F401 - register Gymnasium environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluar una politica entrenada")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best/best_model.zip"),
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default=None)
    return parser.parse_args()


def load_model(model_path: Path, algorithm: str | None, device: str):
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo '{model_path}'. Ejecuta el entrenamiento primero."
        )

    if algorithm is None:
        if "sac" in model_path.name.lower():
            algorithm = "sac"
        else:
            algorithm = "ppo"

    if algorithm == "ppo":
        return PPO.load(model_path, device=device)
    return SAC.load(model_path, device=device)


def main() -> None:
    args = parse_args()
    model = load_model(args.model, args.algorithm, args.device)
    render_mode = "human" if args.render else None
    env = gym.make("BipedWalk-v0", render_mode=render_mode)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_distances: list[float] = []

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
            if args.render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_distances.append(float(info.get("distance", 0.0)))
        print(
            f"Episodio {episode + 1}: recompensa={episode_reward:.2f}, "
            f"pasos={episode_length}, distancia={episode_distances[-1]:.2f} m"
        )

    print(
        "Resumen: "
        f"recompensa media={sum(episode_rewards) / len(episode_rewards):.2f}, "
        f"pasos medios={sum(episode_lengths) / len(episode_lengths):.1f}, "
        f"distancia media={sum(episode_distances) / len(episode_distances):.2f} m"
    )
    env.close()


if __name__ == "__main__":
    main()

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import robot_walk  # noqa: F401 - register Gymnasium environment


def make_env(seed: int, rank: int = 0) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make("BipedWalk-v0")
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def make_vector_env(seed: int, n_envs: int) -> DummyVecEnv | SubprocVecEnv:
    env_fns = [make_env(seed, rank) for rank in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenar el robot bipedo con Stable-Baselines3"
    )
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_envs < 1:
        raise ValueError("--n-envs debe ser al menos 1")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    env = make_vector_env(args.seed, args.n_envs)
    eval_env = DummyVecEnv([make_env(args.seed + 10_000)])

    policy_kwargs = {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]}
        if args.algorithm == "ppo"
        else [256, 256]
    }

    if args.resume is not None:
        if args.algorithm == "ppo":
            model = PPO.load(
                args.resume,
                env=env,
                device=args.device,
                tensorboard_log=str(args.log_dir),
            )
        else:
            model = SAC.load(
                args.resume,
                env=env,
                device=args.device,
                tensorboard_log=str(args.log_dir),
            )
    elif args.algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            device=args.device,
            learning_rate=3e-4,
            n_steps=max(2048 // args.n_envs, 128),
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=1,
            tensorboard_log=str(args.log_dir),
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            seed=args.seed,
            device=args.device,
            learning_rate=3e-4,
            buffer_size=300_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(args.log_dir),
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.model_dir / "best"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=str(args.model_dir / "checkpoints"),
        name_prefix=f"{args.algorithm}_biped",
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback])

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=args.resume is None,
    )

    final_model_path = args.model_dir / f"{args.algorithm}_biped.zip"
    model.save(str(final_model_path.with_suffix("")))

    metadata = {
        "algorithm": args.algorithm,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "model": str(final_model_path),
    }
    (args.model_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    env.close()
    eval_env.close()
    print(f"Modelo guardado en: {final_model_path}")


if __name__ == "__main__":
    main()

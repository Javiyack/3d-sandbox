from gymnasium.envs.registration import register

from .env import BipedWalkEnv

register(
    id="BipedWalk-v0",
    entry_point="robot_walk.env:BipedWalkEnv",
    max_episode_steps=1000,
)

__all__ = ["BipedWalkEnv"]

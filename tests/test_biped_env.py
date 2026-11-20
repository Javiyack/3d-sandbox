import gymnasium as gym
import numpy as np

import robot_walk  # noqa: F401


def test_environment_reset_and_observation() -> None:
    env = gym.make("BipedWalk-v0", random_target=False)
    observation, info = env.reset(seed=42)

    assert env.observation_space.contains(observation)
    assert env.action_space.shape == (12,)
    assert isinstance(info, dict)
    env.close()


def test_environment_random_steps() -> None:
    env = gym.make("BipedWalk-v0", random_target=False)
    env.reset(seed=7)

    for _ in range(200):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert isinstance(info, dict)
        if terminated or truncated:
            break

    env.close()


def test_environment_rgb_render() -> None:
    env = gym.make("BipedWalk-v0", render_mode="rgb_array", random_target=False)
    env.reset(seed=11)

    frame = env.render()

    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    env.close()

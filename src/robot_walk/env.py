from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class BipedWalkEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    JOINT_NAMES: ClassVar[list[str]] = [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle_roll",
        "left_ankle_pitch",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle_roll",
        "right_ankle_pitch",
    ]

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 5,
        episode_length: int = 1000,
        target_speed: float = 0.8,
        random_target: bool = True,
    ) -> None:
        super().__init__()
        xml_path = Path(__file__).parent / "assets" / "biped.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.episode_length = episode_length
        self.target_speed = float(target_speed)
        self.random_target = random_target
        self.action_dim = len(self.JOINT_NAMES)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        observation_dim = 51
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )

        self._torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._left_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot"
        )
        self._right_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot"
        )
        self._root_qposadr = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "root"
        )
        if self._root_qposadr < 0:
            raise RuntimeError("El modelo no contiene el joint free 'root'")
        self._root_qposadr = int(self.model.jnt_qposadr[self._root_qposadr])
        self._root_dofadr = 0

        self._joint_qposadr: list[int] = []
        self._joint_dofadr: list[int] = []
        self._joint_limits: list[tuple[float, float]] = []
        for name in self.JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise RuntimeError(f"El modelo no contiene el joint '{name}'")
            qposadr = int(self.model.jnt_qposadr[joint_id])
            dofadr = int(self.model.jnt_dofadr[joint_id])
            limits = self.model.jnt_range[joint_id]
            self._joint_qposadr.append(qposadr)
            self._joint_dofadr.append(dofadr)
            self._joint_limits.append((float(limits[0]), float(limits[1])))

        self._previous_action = np.zeros(self.action_dim, dtype=np.float32)
        self._step_count = 0
        self._episode_target = self.target_speed
        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None

    def _sample_target_speed(self) -> float:
        if not self.random_target:
            return self.target_speed
        return float(self.np_random.uniform(0.5, 1.2))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._episode_target = self._sample_target_speed()
        self._step_count = 0
        self._previous_action.fill(0.0)

        base_qpos = self._root_qposadr
        self.data.qpos[base_qpos : base_qpos + 3] = np.array(
            [
                float(self.np_random.uniform(-0.05, 0.05)),
                float(self.np_random.uniform(-0.05, 0.05)),
                1.05,
            ],
            dtype=np.float64,
        )
        self.data.qpos[base_qpos + 3 : base_qpos + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

        home_positions = {
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": 0.05,
            "left_knee": 0.10,
            "left_ankle_roll": 0.0,
            "left_ankle_pitch": -0.03,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": -0.05,
            "right_knee": 0.10,
            "right_ankle_roll": 0.0,
            "right_ankle_pitch": -0.03,
        }
        for name, qposadr in zip(self.JOINT_NAMES, self._joint_qposadr, strict=True):
            noise = float(self.np_random.uniform(-0.03, 0.03))
            value = home_positions[name] + noise
            low, high = dict(zip(self.JOINT_NAMES, self._joint_limits, strict=True))[name]
            self.data.qpos[qposadr] = float(np.clip(value, low, high))

        for dofadr in self._joint_dofadr:
            self.data.qvel[dofadr] = 0.0
        for index in range(6):
            self.data.qvel[index] = 0.0

        mujoco.mj_forward(self.model, self.data)
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        observation = self._get_observation()
        reward, terminated, info = self._get_reward_and_termination()
        self._step_count += 1
        truncated = self._step_count >= self.episode_length
        reward = float(reward)
        info.update(self._get_info())

        self._previous_action[:] = action
        return observation, reward, terminated, truncated, info

    def _body_velocity_world(self) -> tuple[np.ndarray, np.ndarray]:
        cvel = np.asarray(self.data.cvel[self._torso_id], dtype=np.float64)
        return cvel[:3].copy(), cvel[3:].copy()

    def _body_rotation(self) -> np.ndarray:
        return np.asarray(
            self.data.xmat[self._torso_id].reshape(3, 3),
            dtype=np.float64,
        )

    def _get_observation(self) -> np.ndarray:
        rotation = self._body_rotation()
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])

        angular_world, linear_world = self._body_velocity_world()
        linear_torso = rotation.T @ linear_world
        angular_torso = rotation.T @ angular_world

        joint_positions: list[float] = []
        joint_velocities: list[float] = []
        joint_limits_by_name = dict(
            zip(self.JOINT_NAMES, self._joint_limits, strict=True)
        )
        for name, qposadr, dofadr in zip(
            self.JOINT_NAMES,
            self._joint_qposadr,
            self._joint_dofadr,
            strict=True,
        ):
            low, high = joint_limits_by_name[name]
            midpoint = 0.5 * (low + high)
            half_range = max(0.5 * (high - low), 1e-6)
            joint_positions.append(
                float(np.clip((self.data.qpos[qposadr] - midpoint) / half_range, -2.0, 2.0))
            )
            joint_velocities.append(
                float(np.clip(self.data.qvel[dofadr] / 8.0, -2.0, 2.0))
            )

        height = float(self.data.xpos[self._torso_id, 2])
        left_foot_height = float(self.data.xpos[self._left_foot_id, 2])
        right_foot_height = float(self.data.xpos[self._right_foot_id, 2])

        phase = 2.0 * np.pi * (self._step_count / 60.0)
        phase_features = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

        observations = np.concatenate(
            [
                np.clip(projected_gravity, -2.0, 2.0).astype(np.float32),
                np.clip(linear_torso / 3.0, -2.0, 2.0).astype(np.float32),
                np.clip(angular_torso / 4.0, -2.0, 2.0).astype(np.float32),
                np.asarray(joint_positions, dtype=np.float32),
                np.asarray(joint_velocities, dtype=np.float32),
                self._previous_action,
                phase_features,
                np.array([np.clip(height / 1.5, 0.0, 2.0)], dtype=np.float32),
                np.array(
                    [np.clip(self._episode_target / 2.0, 0.0, 1.0)],
                    dtype=np.float32,
                ),
                np.array(
                    [
                        np.clip(left_foot_height / 1.5, 0.0, 1.0),
                        np.clip(right_foot_height / 1.5, 0.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return observations.astype(np.float32)

    def _get_reward_and_termination(
        self,
    ) -> tuple[float, bool, dict[str, Any]]:
        rotation = self._body_rotation()
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        tilt = float(np.linalg.norm(projected_gravity[:2]))

        _, linear_world = self._body_velocity_world()
        linear_torso = rotation.T @ linear_world
        forward_velocity = float(linear_torso[0])
        lateral_velocity = float(linear_torso[1])
        angular_world, _ = self._body_velocity_world()
        angular_torso = rotation.T @ angular_world
        yaw_velocity = float(angular_torso[2])

        height = float(self.data.xpos[self._torso_id, 2])
        out_of_bounds = (
            abs(float(self.data.xpos[self._torso_id, 0])) > 20.0
            or abs(float(self.data.xpos[self._torso_id, 1])) > 8.0
        )
        fallen = height < 0.48 or tilt > 0.85 or out_of_bounds

        if fallen:
            reward = -5.0
            info = {
                "fallen": True,
                "forward_velocity": forward_velocity,
                "height": height,
                "tilt": tilt,
            }
            return reward, True, info

        speed_error = (forward_velocity - self._episode_target) / 0.35
        forward_reward = float(np.exp(-0.5 * speed_error**2))
        upright_reward = float(np.exp(-4.0 * tilt**2))
        height_reward = float(np.exp(-((height - 1.05) / 0.15) ** 2))

        action = np.asarray(self.data.ctrl, dtype=np.float32)
        action_energy = float(np.mean(action**2))
        action_delta = float(np.mean((action - self._previous_action) ** 2))

        left_foot_height = float(self.data.xpos[self._left_foot_id, 2])
        right_foot_height = float(self.data.xpos[self._right_foot_id, 2])
        phase = 2.0 * np.pi * (self._step_count / 60.0)
        left_swing = max(float(np.sin(phase)), 0.0)
        right_swing = max(float(-np.sin(phase)), 0.0)
        left_target = 0.11 + 0.10 * left_swing
        right_target = 0.11 + 0.10 * right_swing
        gait_reward = 0.05 * (
            float(np.exp(-((left_foot_height - left_target) / 0.08) ** 2))
            + float(np.exp(-((right_foot_height - right_target) / 0.08) ** 2))
        )

        reward = (
            1.5 * forward_reward
            + 0.8 * upright_reward
            + 0.5 * height_reward
            + gait_reward
            + 0.25
            - 0.08 * min(abs(lateral_velocity), 1.5)
            - 0.04 * min(abs(yaw_velocity), 2.0)
            - 0.002 * action_energy
            - 0.01 * action_delta
        )
        info = {
            "fallen": False,
            "forward_velocity": forward_velocity,
            "target_speed": self._episode_target,
            "height": height,
            "tilt": tilt,
        }
        return reward, False, info

    def _get_info(self) -> dict[str, Any]:
        height = float(self.data.xpos[self._torso_id, 2])
        x_position = float(self.data.xpos[self._torso_id, 0])
        return {
            "distance": x_position,
            "height": height,
            "target_speed": self._episode_target,
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mujoco_viewer

                self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
                self._viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.8])
                self._viewer.cam.distance = 3.2
                self._viewer.cam.elevation = -12.0
                self._viewer.cam.azimuth = 90.0
            self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model,
                    width=640,
                    height=480,
                )
            self._renderer.update_scene(
                self.data,
                camera=-1,
            )
            return np.asarray(self._renderer.render(), dtype=np.uint8)

        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

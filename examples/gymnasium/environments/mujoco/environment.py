"""MuJoCo Environments from Gymnasium wrapped for Containerl."""

from typing import Any

import gymnasium as gym
import numpy as np

from containerl import (
    AllowedInfoValueTypes,
    AllowedTypes,
    CRLEnvironment,
    create_environment_server,
    process_info,
)


class Environment(CRLEnvironment[np.ndarray]):
    """
    Gymnasium MuJoCo Environment Wrapper.

    Available Envs:
        - "Ant-v5"
        - "HalfCheetah-v5"
        - "Hopper-v5"
        - "HumanoidStandup-v5"
        - "Humanoid-v5"
        - "InvertedDoublePendulum-v5"
        - "InvertedPendulum-v5"
        - "Pusher-v5"
        - "Reacher-v5"
        - "Swimmer-v5"
        - "Walker2d-v5"
    """

    def __init__(self) -> None:
        self.render_mode = "rgb_array"
        self._env: gym.Env[np.ndarray, np.ndarray] = gym.make(  # pyright: ignore[reportUnknownMemberType]
            "Ant-v5", render_mode=self.render_mode
        )

        self.observation_space = gym.spaces.Dict(
            {"observation": self._env.observation_space}
        )

        self.action_space = self._env.action_space

    def _process_observation(self, obs: np.ndarray) -> dict[str, AllowedTypes]:
        return {"observation": obs}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, AllowedTypes], dict[str, AllowedInfoValueTypes]]:
        """Reset the environment."""
        obs, info = self._env.reset(seed=seed, options=options)
        return self._process_observation(obs), process_info(info)

    def step(
        self, action: np.ndarray
    ) -> tuple[
        dict[str, AllowedTypes], float, bool, bool, dict[str, AllowedInfoValueTypes]
    ]:
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self._env.step(action)
        return (
            self._process_observation(obs),
            float(reward),
            terminated,
            truncated,
            process_info(info),
        )

    def render(self) -> np.ndarray:  # type: ignore[override]
        """Render the environment."""
        return self._env.render()  # type: ignore

    def close(self) -> None:
        """Close the environment."""
        return self._env.close()  # type: ignore


if __name__ == "__main__":
    create_environment_server(Environment)

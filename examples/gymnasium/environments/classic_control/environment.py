"""Classic Control Environments from Gymnasium wrapped for Containerl."""

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from containerl import (
    AllowedInfoValueTypes,
    AllowedTypes,
    CRLEnvironment,
    create_environment_server,
    process_info,
)


class Environment(CRLEnvironment[np.integer[Any]]):
    """
    Gymnasium Classic Control Environment Wrapper.

    Available Envs:
        - "Acrobot-v1"
        - "CartPole-v1"
        - "MountainCar-v0"
        - "MountainCarContinuous-v0"
        - "Pendulum-v1"
    """

    def __init__(self) -> None:
        self.render_mode = "rgb_array"
        self._env: gym.Env[np.ndarray, np.integer[Any]] = gym.make(  # pyright: ignore[reportUnknownMemberType]
            "CartPole-v1", render_mode=self.render_mode
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
        self, action: np.integer[Any]
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

    def render(self) -> NDArray[np.uint8]:
        """Render the environment and return an RGB array."""
        return self._env.render()  # type: ignore

    def close(self) -> None:
        """Close the environment."""
        return self._env.close()  # type: ignore


if __name__ == "__main__":
    create_environment_server(Environment)

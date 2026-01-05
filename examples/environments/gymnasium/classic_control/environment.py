"""Classic Control Environments from Gymnasium wrapped for Containerl."""

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from containerl import (
    AllowedInfoValueTypes,
    AllowedTypes,
    create_environment_server,
    process_info,
)


class Environment(gym.Env[dict[str, AllowedTypes], AllowedTypes]):
    """
    Gymnasium Classic Control Environment Wrapper.

    Available Envs:
        - "Acrobot-v1"
        - "CartPole-v1"
        - "MountainCar-v0"
        - "MountainCarContinuous-v0"
        - "Pendulum-v1"
    """

    def __init__(self, render_mode: str, env_name: str) -> None:
        self.render_mode = render_mode
        self._env: gym.Env[np.ndarray, AllowedTypes] = gym.make(  # pyright: ignore[reportUnknownMemberType]
            env_name, render_mode=self.render_mode
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
        self, action: AllowedTypes
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
        return self._env.render()  # pyright:ignore[reportReturnType,reportUnknownVariableType]

    def close(self) -> None:
        """Close the environment."""
        return self._env.close()


if __name__ == "__main__":
    create_environment_server(Environment)

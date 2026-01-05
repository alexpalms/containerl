"""Environment wrapper for Atari environments using Gymnasium and ALE."""

from typing import Any

import ale_py
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
    """Environment wrapper for Atari environments using Gymnasium and ALE."""

    def __init__(self, render_mode: str, env_name: str, obs_type: str) -> None:
        # Available Envs: https://ale.farama.org/environments/
        self.render_mode = render_mode
        gym.register_envs(ale_py)
        self._env: gym.Env[np.ndarray, AllowedTypes] = gym.make(  # pyright:ignore[reportUnknownMemberType]
            f"{env_name}", render_mode=self.render_mode, obs_type=obs_type
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
        """Render the environment."""
        return self._env.render()  # pyright:ignore[reportReturnType,reportUnknownVariableType]

    def close(self) -> None:
        """Close the environment."""
        return self._env.close()


if __name__ == "__main__":
    create_environment_server(Environment)

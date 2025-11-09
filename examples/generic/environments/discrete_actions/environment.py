"""A simple discrete action environment with dictionary observations."""

import random
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from containerl.interface import create_environment_server
from containerl.interface.utils import AllowedInfoValueTypes, AllowedTypes


class Environment(gym.Env[dict[str, AllowedTypes], np.integer[Any]]):
    """A simple discrete action environment with dictionary observations."""

    def __init__(self) -> None:
        # Observations are dictionaries
        self.observation_space = spaces.Dict(
            {
                "box": spaces.Box(low=0.0, high=10.0, shape=(10,), dtype=np.float32),
                "box_2": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
                "discrete": spaces.Discrete(4),
                "multi_discrete": spaces.MultiDiscrete([10, 10, 10, 10]),
                "multi_binary": spaces.MultiBinary(4),
                "multi_binary_2": spaces.MultiBinary([2, 2]),
            }
        )

        self.action_space = spaces.Discrete(4)

        self.render_mode = "rgb_array"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, AllowedTypes], dict[str, AllowedInfoValueTypes]]:
        """Reset the environment."""
        super(type(self), self).reset(seed=seed)
        self.env_step = 0
        return self._get_observation(), self._get_info()

    def step(
        self, action: np.integer[Any]
    ) -> tuple[
        dict[str, AllowedTypes], float, bool, bool, dict[str, AllowedInfoValueTypes]
    ]:
        """Take a step in the environment."""
        if not self.action_space.contains(action):
            raise Exception(f"Action {action} not in action space {self.action_space}")

        self.env_step += 1
        return (
            self._get_observation(),
            self._get_reward(),
            self._get_episode_termination(),
            self._get_episode_abortion(),
            self._get_info(),
        )

    def _get_observation(self) -> dict[str, AllowedTypes]:
        return self.observation_space.sample()

    def _get_reward(self) -> float:
        return random.uniform(-1.0, 1.0)  # noqa: S311  # Ignoring criptographically weak RNG for example

    def _get_episode_termination(self) -> bool:
        return True if self.env_step == 10 else False

    def _get_episode_abortion(self) -> bool:
        return False

    def _get_info(self) -> dict[str, AllowedInfoValueTypes]:
        return {}

    def render(self) -> np.ndarray:  # type: ignore[override]
        """Render the environment."""
        # Generate random RGB image
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def close(self) -> None:
        """Close the environment."""
        pass


if __name__ == "__main__":
    create_environment_server(
        cast(type[gym.Env[dict[str, AllowedTypes], AllowedTypes]], Environment)
    )

"""Example Gymnasium Box2D Environment Server."""

from typing import Any

import gymnasium as gym
import numpy as np

from containerl import (
    AllowedInfoValueTypes,
    CRLEnvironment,
    create_environment_server,
    process_info,
)


class Environment(CRLEnvironment[dict[str, np.ndarray], int]):
    """
    Gymnasium Box2D Environment Wrapper.

    Available Envs:
        - "BipedalWalker-v3"
        - "CarRacing-v3"
        - "LunarLander-v3"
    """

    def __init__(self) -> None:
        self.render_mode = "rgb_array"
        self._env: gym.Env[np.ndarray, int] = gym.make(  # pyright: ignore[reportUnknownMemberType]
            "LunarLander-v3", render_mode=self.render_mode
        )

        self.observation_space = gym.spaces.Dict(
            {"observation": self._env.observation_space}
        )

        self.action_space = self._env.action_space

    def _process_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        return {"observation": obs}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, AllowedInfoValueTypes]]:
        """Reset the environment."""
        obs, info = self._env.reset(seed=seed, options=options)
        return self._process_observation(obs), process_info(info)

    def step(
        self, action: int
    ) -> tuple[
        dict[str, np.ndarray], float, bool, bool, dict[str, AllowedInfoValueTypes]
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

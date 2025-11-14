"""MuJoCo Environments from Gymnasium wrapped for Containerl."""

from typing import Any, cast

import gymnasium as gym
import numpy as np

from containerl.interface import create_environment_server
from containerl.interface.utils import (
    AllowedInfoBaseTypes,
    AllowedInfoValueTypes,
    AllowedTypes,
)


class Environment(gym.Env[dict[str, np.ndarray], np.ndarray]):
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

    def _process_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        return {"observation": obs}

    def _process_info(self, info: dict[str, Any]) -> dict[str, AllowedInfoValueTypes]:
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                info[key] = value.tolist()
            elif isinstance(value, np.number):  # Catches all numeric types (int, float)
                info[key] = value.item()  # .item() converts to native Python type
            elif isinstance(value, np.bool_):
                info[key] = bool(cast(bool, value))
            elif isinstance(value, (list, tuple)):
                value = cast(list[AllowedInfoBaseTypes], value)
                # Process lists and tuples that might contain numpy types
                processed: list[AllowedInfoValueTypes] = []
                for item in value:
                    if isinstance(item, np.ndarray):
                        processed.append(item.tolist())
                    elif isinstance(item, np.integer) or isinstance(item, np.floating):
                        processed.append(item.item())
                    elif isinstance(item, np.bool_):
                        processed.append(bool(item))
                    else:
                        processed.append(item)
                # Convert back to the original type (list or tuple)
                info[key] = processed

        return info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, AllowedInfoValueTypes]]:
        """Reset the environment."""
        obs, info = self._env.reset(seed=seed, options=options)
        return self._process_observation(obs), self._process_info(info)

    def step(
        self, action: np.ndarray
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
            self._process_info(info),
        )

    def render(self) -> np.ndarray:  # type: ignore[override]
        """Render the environment."""
        return self._env.render()  # type: ignore

    def close(self) -> None:
        """Close the environment."""
        return self._env.close()  # type: ignore


if __name__ == "__main__":
    create_environment_server(
        cast(type[gym.Env[dict[str, AllowedTypes], AllowedTypes]], Environment)
    )

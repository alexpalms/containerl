"""A vectorized environment with discrete action space and dictionary observations."""

from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from containerl import (
    AllowedInfoValueTypes,
    AllowedTypes,
    CRLVecGymEnvironment,
    create_vec_environment_server,
)


class Environment(CRLVecGymEnvironment):
    """A simple multidiscrete action vectorized environment with dictionary observations."""

    def __init__(self, num_envs: int = 1):
        self.num_envs = num_envs
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

        self.action_space = spaces.MultiDiscrete([4, 4, 4, 4])

        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.render_mode = "rgb_array"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[
        dict[str, NDArray[np.floating | np.integer]],
        list[dict[str, AllowedInfoValueTypes]],
    ]:
        """Reset the environment."""
        self.env_step = 0
        return self._get_observation(), self._get_info()

    def step(
        self, action: NDArray[np.floating | np.integer]
    ) -> tuple[
        dict[str, NDArray[np.floating | np.integer]],
        NDArray[np.floating],
        NDArray[np.bool_],
        NDArray[np.bool_],
        list[dict[str, AllowedInfoValueTypes]],
    ]:
        """Take a step in the environment."""
        # Expect action shape to be (num_envs, 4)
        if action.shape != (self.num_envs, 4):
            raise Exception(
                f"Action shape must be ({self.num_envs}, 4), got {action.shape}"
            )
        # Check each action individually
        for i, single_action in enumerate(action):
            if not self.action_space.contains(single_action):
                raise Exception(
                    f"Action {single_action} at index {i} not in action space {self.action_space}"
                )

        self.env_step += 1
        return (
            self._get_observation(),
            self._get_reward(),
            self._get_episode_termination(),
            self._get_episode_abortion(),
            self._get_info(),
        )

    def _get_observation(self) -> dict[str, NDArray[np.floating | np.integer]]:
        obs_list: list[dict[str, AllowedTypes]] = [
            self.single_observation_space.sample() for _ in range(self.num_envs)
        ]
        return {
            space_key: np.stack([obs[space_key] for obs in obs_list])
            for space_key in obs_list[0].keys()
        }

    def _get_reward(self) -> NDArray[np.floating]:
        return np.random.uniform(-1.0, 1.0, size=self.num_envs).astype(np.float32)

    def _get_episode_termination(self) -> NDArray[np.bool_]:
        return np.full(self.num_envs, self.env_step == 10, dtype=np.bool_)

    def _get_episode_abortion(self) -> NDArray[np.bool_]:
        return np.zeros(self.num_envs, dtype=np.bool_)

    def _get_info(self) -> list[dict[str, AllowedInfoValueTypes]]:
        return [{} for _ in range(self.num_envs)]

    def render(self) -> NDArray[np.uint8]:
        """Render the environment."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def close(self) -> None:
        """Close the environment."""
        pass


if __name__ == "__main__":
    create_vec_environment_server(Environment)

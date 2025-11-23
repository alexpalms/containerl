"""gRPC server factory for Gymnasium environments."""

# gRPC Server Implementation
from abc import ABC, abstractmethod
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from ..utils import (
    AllowedInfoValueTypes,
    AllowedTypes,
)


class CRLEnvironmentBase(ABC):
    """Abstract base class for Environments."""

    _observation_space: gym.spaces.Space[dict[str, AllowedTypes]]
    _action_space: gym.spaces.Space[AllowedTypes]
    _render_mode: str | None = None
    _np_random: np.random.Generator

    @abstractmethod
    def init(self, **init_args: Any) -> None:
        """Initialize the environment."""
        pass

    def observation_space(self) -> gym.spaces.Space[dict[str, AllowedTypes]]:
        """Get the observation space of the environment."""
        return self._observation_space

    def action_space(self) -> gym.spaces.Space[AllowedTypes]:
        """Get the action space of the environment."""
        return self._action_space

    def render_mode(self) -> str | None:
        """Get the render mode of the environment."""
        return self._render_mode

    def np_random(self) -> np.random.Generator:
        """Get the action space of the environment."""
        return self._np_random

    @abstractmethod
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[
        dict[str, AllowedTypes],
        dict[str, AllowedInfoValueTypes],
    ]:
        """Reset the environment."""
        pass

    @abstractmethod
    def step(
        self, action: AllowedTypes
    ) -> tuple[
        dict[str, AllowedTypes],
        SupportsFloat,
        bool,
        bool,
        dict[str, AllowedInfoValueTypes],
    ]:
        """Take a step in the environment."""
        pass

    @abstractmethod
    def render(self) -> NDArray[np.uint8] | None:
        """Render the environment and return an image as a numpy array if applicable."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the environment."""
        pass


class CRLGymEnvironmentAdapter(CRLEnvironmentBase):
    """Abstract base class for Environments."""

    def __init__(
        self,
        gym_class: type[gym.Env[dict[str, AllowedTypes], AllowedTypes]],
    ) -> None:
        self._env_class: type[gym.Env[dict[str, AllowedTypes], AllowedTypes]] = (
            gym_class
        )

    def init(self, **init_args: Any) -> None:
        """Initialize the environment."""
        self._env: gym.Env[dict[str, AllowedTypes], AllowedTypes] = self._env_class(
            **init_args
        )
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        self._render_mode = self._env.render_mode
        self._np_random = self._env.np_random

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[
        dict[str, AllowedTypes],
        dict[str, AllowedInfoValueTypes],
    ]:
        """Reset the environment."""
        return self._env.reset(seed=seed, options=options)

    def step(
        self, action: AllowedTypes
    ) -> tuple[
        dict[str, AllowedTypes],
        SupportsFloat,
        bool,
        bool,
        dict[str, AllowedInfoValueTypes],
    ]:
        """Take a step in the environment."""
        return self._env.step(action)

    def render(self) -> NDArray[np.uint8] | None:
        """Render the environment and return an image as a numpy array if applicable."""
        return self._env.render()  # type: ignore[return-value]

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

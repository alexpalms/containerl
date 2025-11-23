"""gRPC server factory for Gymnasium environments."""

# gRPC Server Implementation
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from numpy.typing import NDArray

from ..utils import AllowedInfoValueTypes, AllowedTypes


class CRLVecEnvironmentBase(ABC):
    """Abstract base class for Vectorized Environments."""

    _num_envs: int
    _single_observation_space: spaces.Space[dict[str, AllowedTypes]]
    _single_action_space: spaces.Space[AllowedTypes]
    _observation_space: spaces.Space[dict[str, NDArray[np.floating | np.integer[Any]]]]
    _action_space: spaces.Space[NDArray[np.floating | np.integer[Any]]]
    _render_mode: str | None = None
    _np_random: np.random.Generator

    @abstractmethod
    def init(self, **init_args: Any) -> None:
        """Initialize the environment."""
        pass

    def single_observation_space(self) -> spaces.Space[dict[str, AllowedTypes]]:
        """Get the observation space of the environment."""
        return self._single_observation_space

    def single_action_space(self) -> spaces.Space[AllowedTypes]:
        """Get the action space of the environment."""
        return self._single_action_space

    def observation_space(
        self,
    ) -> spaces.Space[dict[str, NDArray[np.floating | np.integer[Any]]]]:
        """Get the observation space of the environment."""
        return self._observation_space

    def action_space(self) -> spaces.Space[NDArray[np.floating | np.integer[Any]]]:
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
        self,
        *,
        seed: NDArray[np.integer[Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[
        dict[str, NDArray[np.floating | np.integer[Any]]],
        list[dict[str, AllowedInfoValueTypes]],
    ]:
        """Reset the environment."""
        pass

    @abstractmethod
    def step(
        self, action: NDArray[np.floating | np.integer[Any]]
    ) -> tuple[
        dict[str, NDArray[np.floating | np.integer[Any]]],
        NDArray[np.floating],
        NDArray[np.bool_],
        NDArray[np.bool_],
        list[dict[str, AllowedInfoValueTypes]],
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


class CRLGymVecEnvironmentAdapter(CRLVecEnvironmentBase):
    """Abstract base class for Environments."""

    def __init__(
        self,
        gym_class: type[VectorEnv],
    ) -> None:
        self._env_class = gym_class

    def init(self, **init_args: Any) -> None:
        """Initialize the environment."""
        self._env = self._env_class(**init_args)
        self._num_envs = self._env.num_envs

        if not isinstance(self._env.single_observation_space, spaces.Dict):
            raise Exception(
                "CRLGymVecEnvironmentAdapter only supports Dict observation spaces."
            )
        if not isinstance(
            self._env.single_action_space,
            (
                spaces.Discrete,
                spaces.Box,
                spaces.MultiBinary,
                spaces.MultiDiscrete,
            ),
        ):
            raise Exception(
                "CRLGymVecEnvironmentAdapter only supports Discrete, Box, MultiBinary, and MultiDiscrete action spaces."
            )
        self._single_observation_space = self._env.single_observation_space
        self._single_action_space = self._env.single_action_space
        self._observation_space = cast(
            spaces.Space[dict[str, NDArray[np.floating | np.integer[Any]]]],
            self._env.observation_space,
        )
        self._action_space = cast(
            spaces.Space[NDArray[np.floating | np.integer[Any]]],
            self._env.action_space,
        )
        self._render_mode = self._env.render_mode
        self._np_random = self._env.np_random

    def reset(
        self,
        *,
        seed: NDArray[np.integer[Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[
        dict[str, NDArray[np.floating | np.integer[Any]]],
        list[dict[str, AllowedInfoValueTypes]],
    ]:
        """Reset the environment."""
        return self._env.reset(seed=seed, options=options)

    def step(
        self, action: NDArray[np.floating | np.integer[Any]]
    ) -> tuple[
        dict[str, NDArray[np.floating | np.integer[Any]]],
        NDArray[np.floating],
        NDArray[np.bool_],
        NDArray[np.bool_],
        list[dict[str, AllowedInfoValueTypes]],
    ]:
        """Take a step in the environment."""
        return self._env.step(action)

    def render(self) -> NDArray[np.uint8] | None:
        """Render the environment and return an image as a numpy array if applicable."""
        return self._env.render()  # type: ignore[return-value]

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

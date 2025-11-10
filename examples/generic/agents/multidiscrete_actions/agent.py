"""An example of a generic agent with multidiscrete action space and dictionary observations."""

from typing import cast

import numpy as np
from gymnasium import spaces

from containerl.interface import create_agent_server
from containerl.interface.utils import Agent as BaseAgent
from containerl.interface.utils import AllowedTypes


class Agent(BaseAgent[dict[str, AllowedTypes], np.ndarray]):
    """A simple agent with dictionary observations and multidiscrete action space."""

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

        self.action_space = spaces.MultiDiscrete([4, 4, 4, 4])

    def get_action(self, observation: dict[str, AllowedTypes]) -> np.ndarray:
        """Return a random action based on the observation."""
        for key, space in cast(spaces.Dict, self.observation_space).spaces.items():
            if key not in observation:
                raise Exception("Missing observation key: {key}")
            if not space.contains(observation[key]):
                raise Exception(
                    f"Observation {observation[key]} not in space {space} for key {key}"
                )
        return self.action_space.sample()


if __name__ == "__main__":
    agent = Agent()
    create_agent_server(cast(BaseAgent[dict[str, AllowedTypes], AllowedTypes], agent))

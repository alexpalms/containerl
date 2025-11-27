"""An example of a generic agent with continuous action space and dictionary observations."""

from typing import cast

import numpy as np
from gymnasium import spaces

from containerl import AllowedTypes, CRLAgent, create_agent_server


class Agent(CRLAgent):
    """A simple agent with dictionary observations and continuous action space."""

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

        self.action_space = spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32)

    def get_action(self, observation: dict[str, AllowedTypes]) -> AllowedTypes:
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
    create_agent_server(Agent)

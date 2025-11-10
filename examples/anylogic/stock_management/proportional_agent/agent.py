"""Simple proportional Agent for Anylogic stock problem."""

from typing import cast

import numpy as np
from gymnasium import spaces

from containerl.interface import create_agent_server
from containerl.interface.utils import Agent as BaseAgent
from containerl.interface.utils import AllowedTypes


class Agent(BaseAgent[dict[str, AllowedTypes], np.ndarray]):
    """Simple proportional Agent for Anylogic stock problem."""

    def __init__(self) -> None:
        self.target_stock = 5000
        self.proportional_constant = 0.1
        self.observation_space = spaces.Dict(
            {
                "stock": spaces.Box(
                    low=0.0, high=10_000.0, shape=(1,), dtype=np.float32
                ),
                "order_rate": spaces.Box(
                    low=0.0, high=50.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        self.action_space = spaces.Box(0, 50, shape=(1,), dtype=np.float32)

    def get_action(self, observation: dict[str, AllowedTypes]) -> np.ndarray:
        """Return the optimal action as calculated by the proportional controller."""
        act = np.clip(
            self.proportional_constant * (self.target_stock - observation["stock"]),
            0,
            50,
        )
        return act


if __name__ == "__main__":
    agent = Agent()
    create_agent_server(cast(BaseAgent[dict[str, AllowedTypes], AllowedTypes], agent))

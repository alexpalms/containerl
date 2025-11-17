"""Simple proportional Agent for Anylogic stock problem."""

import numpy as np
from gymnasium import spaces

from containerl import AllowedTypes, create_agent_server
from containerl import CRLAgent as BaseAgent


class Agent(BaseAgent[np.ndarray]):
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
    create_agent_server(agent)

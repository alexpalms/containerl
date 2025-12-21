"""Simple proportional Agent for Anylogic stock problem."""

import numpy as np
from gymnasium import spaces

from containerl import (
    AllowedInfoValueTypes,
    AllowedTypes,
    CRLAgent,
    create_agent_server,
)


class Agent(CRLAgent):
    """Simple proportional Agent for Anylogic stock problem."""

    def __init__(self, target_stock: float, proportional_constant: float) -> None:
        self.target_stock = target_stock
        self.proportional_constant = proportional_constant
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
        self.init_info: dict[str, AllowedInfoValueTypes] = {
            "target_stock": target_stock,
            "proportional_constant": proportional_constant,
        }

    def get_action(self, observation: dict[str, AllowedTypes]) -> AllowedTypes:
        """Return the optimal action as calculated by the proportional controller."""
        act = np.clip(
            self.proportional_constant * (self.target_stock - observation["stock"]),
            0,
            50,
        )
        return act


if __name__ == "__main__":
    create_agent_server(Agent)

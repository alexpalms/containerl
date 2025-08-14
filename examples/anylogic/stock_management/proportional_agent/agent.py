import numpy as np
from gymnasium import spaces

from containerl.interface import create_agent_server

class Agent:
    def __init__(self):
        self.target_stock = 5000
        self.proportional_constant = 0.1
        self.observation_space = spaces.Dict(
            {
                "stock": spaces.Box(low=0.0, high=10_000.0, shape=(1,), dtype=np.float32),
                "order_rate": spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32),
            }
        )

        self.action_space = spaces.Box(0, 50, shape=(1,), dtype=np.float32)


    def get_action(self, obs):
        act = np.clip(self.proportional_constant * (self.target_stock - obs["stock"]), 0, 50)
        return act

if __name__ == "__main__":
    agent = Agent()
    create_agent_server(agent)
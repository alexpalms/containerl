import numpy as np
from gymnasium import spaces

from containerl.interface import create_agent_server

class Agent:
    def __init__(self):
        # Observations are dictionaries
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=-10.0, high=10.0, shape=(45,), dtype=np.float32),
            }
        )

        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(12,), dtype=np.float32)

    def get_action(self, observation):
        # Check if observation is a valid instance of the observation space
        assert isinstance(observation, dict), "Observation must be a dictionary"
        for key, space in self.observation_space.spaces.items():
            assert key in observation, f"Missing observation key: {key}"
            assert space.contains(observation[key]), f"Observation {observation[key]} not in space {space} for key {key}"
        return self.action_space.sample()

if __name__ == "__main__":
    agent = Agent()
    create_agent_server(agent)
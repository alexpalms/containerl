import numpy as np
import random
from gymnasium import spaces
import gymnasium as gym

from containerl.interface import create_environment_server

class Environment(gym.Env):
    def __init__(self):
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

        self.action_space = spaces.Discrete(4)

        self.render_mode = "rgb_array"

    def reset(self, seed=None, options=None):
        super(type(self), self).reset(seed=seed)
        self.env_step = 0
        return self.get_observation(), self.get_info()

    def step(self, action):
        assert self.action_space.contains(action), f"Action {action} not in action space {self.action_space}"
        self.env_step += 1
        return self.get_observation(), self.get_reward(), self.get_episode_termination(), self.get_episode_abortion(), self.get_info()

    def get_observation(self):
        return self.observation_space.sample()

    def get_reward(self):
        return random.uniform(-1.0, 1.0)

    def get_episode_termination(self):
        return True if self.env_step == 10 else False

    def get_episode_abortion(self):
        return False

    def get_info(self):
        return {}

    def render(self):
        # Generate random RGB image
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def close(self):
        pass

if __name__ == "__main__":
    create_environment_server(Environment)
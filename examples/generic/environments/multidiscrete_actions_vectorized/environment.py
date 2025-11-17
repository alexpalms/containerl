import gymnasium as gym
import numpy as np
from gymnasium import spaces

from containerl import create_environment_server


class Environment(gym.Env):
    def __init__(self, num_envs=1):
        self.num_envs = num_envs
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

        self.render_mode = "rgb_array"

    def reset(self, seed=None, options=None):
        super(type(self), self).reset(seed=seed)
        self.env_step = 0
        return self.get_observation(), self.get_info()

    def step(self, action):
        # Expect action shape to be (num_envs, 1)
        assert isinstance(action, np.ndarray), (
            f"Action must be numpy array, got {type(action)}"
        )
        assert action.shape == (self.num_envs, *self.action_space.shape), (
            f"Action shape must be ({self.num_envs}, {self.action_space.shape}), got {action.shape}"
        )
        # Check each action individually
        for i, single_action in enumerate(action):
            assert self.action_space.contains(single_action), (
                f"Action {single_action} at index {i} not in action space {self.action_space}"
            )

        self.env_step += 1
        return (
            self.get_observation(),
            self.get_reward(),
            self.get_episode_termination(),
            self.get_episode_abortion(),
            self.get_info(),
        )

    def get_observation(self):
        return {
            space_key: np.stack([space.sample() for _ in range(self.num_envs)])
            for space_key, space in self.observation_space.spaces.items()
        }

    def get_reward(self):
        return np.random.uniform(-1.0, 1.0, size=self.num_envs).astype(np.float32)

    def get_episode_termination(self):
        return np.full(self.num_envs, self.env_step == 10, dtype=bool)

    def get_episode_abortion(self):
        return np.zeros(self.num_envs, dtype=bool)

    def get_info(self):
        return [{} for _ in range(self.num_envs)]

    def render(self):
        # Generate random RGB image
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def close(self):
        pass


if __name__ == "__main__":
    create_environment_server(Environment)

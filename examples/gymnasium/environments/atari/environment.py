import gymnasium as gym
import ale_py
import numpy as np
from containerl.interface import create_environment_server

class Environment(gym.Env):
    def __init__(self):
        """
        Available Envs: https://ale.farama.org/environments/
        """
        self.render_mode = "rgb_array"
        gym.register_envs(ale_py)
        self._env = gym.make("ALE/Breakout-v5", render_mode=self.render_mode, obs_type="ram")

        self.is_observation_dict = isinstance(self._env.observation_space, gym.spaces.Dict)
        if not self.is_observation_dict:
            self.observation_space = gym.spaces.Dict({"observation": self._env.observation_space})
        else:
            self.observation_space = self._env.observation_space

        self.action_space = self._env.action_space

    def _process_observation(self, obs):
        if self.is_observation_dict:
            return obs
        else:
            return {"observation": obs}

    def _process_info(self, info):
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                info[key] = value.tolist()
            elif isinstance(value, np.number):  # Catches all numeric types (int, float)
                info[key] = value.item()  # .item() converts to native Python type
            elif isinstance(value, np.bool_):
                info[key] = bool(value)
            elif isinstance(value, (list, tuple)):
                # Process lists and tuples that might contain numpy types
                processed = []
                for item in value:
                    if isinstance(item, np.ndarray):
                        processed.append(item.tolist())
                    elif isinstance(item, np.number):
                        processed.append(item.item())
                    elif isinstance(item, np.bool_):
                        processed.append(bool(item))
                    else:
                        processed.append(item)
                # Convert back to the original type (list or tuple)
                info[key] = type(value)(processed)

        return info

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return self._process_observation(obs), self._process_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._process_observation(obs), reward, terminated, truncated, self._process_info(info)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

if __name__ == "__main__":
    create_environment_server(Environment)
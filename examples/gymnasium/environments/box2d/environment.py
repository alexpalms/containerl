import gymnasium as gym

from containerl.interface import create_environment_server

class Environment(gym.Env):
    def __init__(self):
        """
        Available Envs:
         - "BipedalWalker-v3"
         - "CarRacing-v3"
         - "LunarLander-v3"
        """
        self.render_mode = "rgb_array"
        self._env = gym.make("LunarLander-v3", render_mode=self.render_mode)

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

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return self._process_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._process_observation(obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

if __name__ == "__main__":
    create_environment_server(Environment)
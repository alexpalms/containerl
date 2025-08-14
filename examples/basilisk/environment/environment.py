import gymnasium as gym
import numpy as np
from bsk_rl import act, data, obs, scene, sats
from bsk_rl.sim import dyn, fsw
from containerl.interface import create_environment_server

class Environment(gym.Env):
    def __init__(self):
        self.render_mode = None

        class MyScanningSatellite(sats.AccessSatellite):
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction")
                ),
                obs.Eclipse(),
            ]
            action_spec = [
                act.Scan(duration=60.0),  # Scan for 1 minute
                act.Charge(duration=600.0),  # Charge for 10 minutes
            ]
            dyn_type = dyn.ContinuousImagingDynModel
            fsw_type = fsw.ContinuousImagingFSWModel


        MyScanningSatellite.default_sat_args()

        sat_args = {}

        # Set some parameters as constants
        sat_args["imageAttErrorRequirement"] = 0.05
        sat_args["dataStorageCapacity"] = 1e10
        sat_args["instrumentBaudRate"] = 1e7
        sat_args["storedCharge_Init"] = 50000.0

        # Randomize the initial storage level on every reset
        sat_args["storageInit"] = lambda: np.random.uniform(0.25, 0.75) * 1e10

        # Make the satellite
        sat = MyScanningSatellite(name="EO1", sat_args=sat_args)

        self._env = gym.make(
            "SatelliteTasking-v1",
            satellite=sat,
            scenario=scene.UniformNadirScanning(),
            rewarder=data.ScanningTimeReward(),
            time_limit=5700.0,  # approximately 1 orbit
            log_level="INFO",
        )

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
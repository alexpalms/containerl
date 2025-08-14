import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.constants import backend as gs_backend
from gymnasium import spaces
import numpy as np

from containerl.interface import create_environment_server

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

ENV_CFG = {
    "num_actions": 4,
    # termination
    "termination_if_roll_greater_than": 180,  # degree
    "termination_if_pitch_greater_than": 180,
    "termination_if_close_to_ground": 0.1,
    "termination_if_x_greater_than": 3.0,
    "termination_if_y_greater_than": 3.0,
    "termination_if_z_greater_than": 2.0,
    # base pose
    "base_init_pos": [0.0, 0.0, 1.0],
    "base_init_quat": [1.0, 0.0, 0.0, 0.0],
    "episode_length_s": 15.0,
    "at_target_threshold": 0.1,
    "resampling_time_s": 3.0,
    "simulate_action_latency": True,
    "clip_actions": 1.0,
    # visualization
    "visualize_target": True,
    "visualize_camera": False,
    "max_visualize_FPS": 60,
}

OBS_CFG = {
    "num_obs": 17,
    "obs_scales": {
        "rel_pos": 1 / 3.0,
        "lin_vel": 1 / 3.0,
        "ang_vel": 1 / 3.14159,
    },
}

REWARD_CFG = {
    "yaw_lambda": -10.0,
    "reward_scales": {
        "target": 10.0,
        "smooth": -1e-4,
        "yaw": 0.01,
        "angular": -2e-4,
        "crash": -10.0,
    },
}

COMMAND_CFG = {
    "num_commands": 3,
    "pos_x_range": [-1.0, 1.0],
    "pos_y_range": [-1.0, 1.0],
    "pos_z_range": [1.0, 1.0],
}


class Environment:
    def __init__(self, num_envs=1, render_mode=None, device="cpu", show_viewer=False):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.render_mode = render_mode
        self.show_viewer = show_viewer
        self.num_obs = OBS_CFG["num_obs"]
        self.num_actions = ENV_CFG["num_actions"]
        self.num_commands = COMMAND_CFG["num_commands"]

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=-10.0, high=10.0, shape=(self.num_obs,), dtype=np.float32),
            }
        )

        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.num_actions,), dtype=np.float32)

        self.simulate_action_latency = ENV_CFG["simulate_action_latency"]
        self.dt = 0.01  # control frequence on real robot is 50hz
        self.max_episode_length = math.ceil(ENV_CFG["episode_length_s"] / self.dt)

        self.env_cfg = ENV_CFG
        self.obs_cfg = OBS_CFG
        self.reward_cfg = REWARD_CFG
        self.command_cfg = COMMAND_CFG

        self.obs_scales = OBS_CFG["obs_scales"]
        self.reward_scales = REWARD_CFG["reward_scales"]
        self.initialized = False
        self.render_mode = "rgb_array"

    def _init_genesis(self, seed: int | None=None):

        # create scene
        gs.init(logging_level="warning", backend=gs_backend.cpu, seed=seed)

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=ENV_CFG["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=self.show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        if self.render_mode is not None:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=False,
            )

        # add drone
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))

        # build
        self.scene.build(n_envs=self.num_envs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.info = [{ "custom_stats": {} }] * self.num_envs  # extra information for logging

    def reset(self, seed=None, options=None):
        if not self.initialized:
            self._init_genesis(seed)
            self.initialized = True
        self.reset_buf[:] = True
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self._get_observations(), self._get_info()

    def step(self, actions):
        self.actions = torch.as_tensor(actions, device=self.device, dtype=gs.tc_float)
        self.actions = torch.clip(self.actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.actions.cpu()
        # exec_actions = self.last_actions.cpu() if self.simulate_action_latency else self.actions.cpu()
        # target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        # self.drone.control_dofs_position(target_dof_pos)

        # 14468 is hover rpm
        self.drone.set_propellels_rpm((1 + exec_actions * 0.8) * 14468.429183500699)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

        # resample commands
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        # check termination and reset
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.time_outs = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.time_outs[time_out_idx] = 1.0

        self._reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        return self._get_observations(), self._get_reward(), self._get_episode_termination(), self._get_episode_abortion(), self._get_info()

    def close(self) -> None:
        gs.destroy()

    def render(self):
        if self.render_mode is None:
            return False
        # render rgb, depth, segmentation mask and normal map
        rgb, depth, segmentation, normal = self.cam.render(depth=True, segmentation=True, normal=True)

        # Create a 2x2 grid
        h, w = rgb.shape[0], rgb.shape[1]
        grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)

        # Place images in the grid
        grid[:h, :w] = rgb

        # Normalize depth dynamically based on min and max values
        depth_min = np.min(depth)
        depth_max = np.max(depth)
        if depth_max > depth_min:  # Avoid division by zero
            depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth, dtype=np.uint8)
        depth_rgb = np.repeat(depth_normalized[:, :, np.newaxis], 3, axis=2)
        grid[:h, w:] = depth_rgb

        # Convert segmentation to a colorful visualization
        # Create a colormap for segmentation (each object gets a distinct color)
        unique_segments = np.unique(segmentation)
        segmentation_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Assign different colors to different segment IDs
        for i, seg_id in enumerate(unique_segments):
            # Create a color based on the segment ID
            r = (seg_id * 37) % 255
            g = (seg_id * 91) % 255
            b = (seg_id * 151) % 255

            mask = segmentation == seg_id
            segmentation_rgb[mask] = [r, g, b]

        grid[h:, :w] = segmentation_rgb

        grid[h:, w:] = normal

        return grid

    def _get_observations(self):
        return {"obs": self.obs_buf.cpu().numpy()}

    def _get_reward(self):
        return self.rew_buf.cpu().numpy()

    def _get_episode_termination(self):
        return self.reset_buf.cpu().numpy()

    def _get_episode_abortion(self):
        # Return boolean array indicating which environments timed out
        return self.time_outs.cpu().numpy() > 0.0

    def _get_info(self):
        return self.info

    def _reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill info
        for key in self.episode_sums.keys():
            for env_idx in envs_idx:
                self.info[env_idx]["custom_stats"].update({ "rew_" + key: self.episode_sums[key][env_idx].cpu().numpy().tolist() })
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)
        if self.target is not None:
            self.target.set_pos(self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx)

    def _at_target(self):
        at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).flatten()
        )
        return at_target

    # ------------ reward functions----------------
    def _reward_target(self):
        return torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        return torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))

    def _reward_angular(self):
        return torch.norm(self.base_ang_vel / 3.14159, dim=1)

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew

if __name__ == "__main__":
    create_environment_server(Environment)

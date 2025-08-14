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
    "num_actions": 12,
    # joint/link names
    "default_joint_angles": {  # [rad]
        "FL_hip_joint": 0.0,
        "FR_hip_joint": 0.0,
        "RL_hip_joint": 0.0,
        "RR_hip_joint": 0.0,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    },
    "dof_names": [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ],
    # PD
    "kp": 20.0,
    "kd": 0.5,
    # termination
    "termination_if_roll_greater_than": 10,  # degree
    "termination_if_pitch_greater_than": 10,
    # base pose
    "base_init_pos": [0.0, 0.0, 0.42],
    "base_init_quat": [1.0, 0.0, 0.0, 0.0],
    "episode_length_s": 20.0,
    "resampling_time_s": 4.0,
    "action_scale": 0.25,
    "simulate_action_latency": True,
    "clip_actions": 100.0,
}

OBS_CFG = {
    "num_obs": 45,
    "obs_scales": {
        "lin_vel": 2.0,
        "ang_vel": 0.25,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
    },
}

REWARD_CFG = {
    "tracking_sigma": 0.25,
    "base_height_target": 0.3,
    "feet_height_target": 0.075,
    "reward_scales": {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.2,
        "lin_vel_z": -1.0,
        "base_height": -50.0,
        "action_rate": -0.005,
        "similar_to_default": -0.1,
    },
}

COMMAND_CFG = {
    "num_commands": 3,
    "lin_vel_x_range": [0.5, 0.5],
    "lin_vel_y_range": [0, 0],
    "ang_vel_range": [0, 0],
}


class Environment:
    def __init__(self, num_envs=1, device="cpu", show_viewer=False):
        self.device = torch.device(device)
        self.num_envs = num_envs
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

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequence on real robot is 50hz
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

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
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
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        if self.render_mode is not None:
            self.cam = self.scene.add_camera(
                res    = (640, 480),
                pos    = (2.0, 0.0, 2.5),
                lookat = (0.0, 0.0, 0.5),
                fov    = 40,
                GUI    = False
            )

        # build
        self.scene.build(n_envs=self.num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
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
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

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
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

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

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill info
        for key in self.episode_sums.keys():
            for env_idx in envs_idx:
                self.info[env_idx]["custom_stats"].update({ "rew_" + key: self.episode_sums[key][env_idx].cpu().numpy().tolist() })
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

if __name__ == "__main__":
    create_environment_server(Environment)
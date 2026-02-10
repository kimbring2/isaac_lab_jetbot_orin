# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pxr import UsdGeom, Sdf
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .isaac_lab_jetbot_orin_env_cfg import IsaacLabJetbotOrinEnvCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.assets import AssetBase, AssetBaseCfg

from isaacsim.core.utils.extensions import enable_extension
#enable_extension("isaacsim.debug_draw")
import isaacsim.util.debug_draw._debug_draw as _debug_draw


class IsaacLabJetbotOrinEnv(DirectRLEnv):
    cfg: IsaacLabJetbotOrinEnvCfg

    def __init__(self, cfg: IsaacLabJetbotOrinEnv, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        # 1. Spawn the track first (it's the floor)
        self.track = RigidObject(self.cfg.track_cfg)
        self.scene.rigid_objects["track"] = self.track

        # 2. Spawn the robot
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # add ground plane
        #spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # add left camera
        left_camera_cfg = CameraCfg(
            prim_path="/World/envs/env_.*/Robot/Body/Camera/jetbot_camera_1",
            update_period=0.0, # 0.0 means every simulation step
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=None,
            offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        )
        self.left_camera = Camera(left_camera_cfg)
        self.scene.sensors["left_camera"] = self.left_camera

        # add right camera
        left_right_cfg = CameraCfg(
            prim_path="/World/envs/env_.*/Robot/Body/Camera/jetbot_camera_0",
            update_period=0.0, # 0.0 means every simulation step
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=None,
            offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        )
        self.right_camera = Camera(left_right_cfg)
        self.scene.sensors["right_camera"] = self.right_camera

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # add road lane
        curve_prim = UsdGeom.BasisCurves.Get(self.sim.stage, "/World/envs/env_0/Track/Road_Lane_1")
        local_points = list(curve_prim.GetPointsAttr().Get())
        self.draw = _debug_draw.acquire_debug_draw_interface()
        
        # add waypoint
        self.debug_waypoints = True
        all_env_points_list = []

        # Scale factor of 1/10
        scale = 0.1

        for origin in self.scene.env_origins:
            # Scale local points by 0.1 AND shift to environment's world position
            world_points = [
                (p[0] * scale + origin[0],  p[1] * scale + origin[1], p[2] * scale + origin[2] -0.2
                ) for p in local_points
            ]

            starts = world_points[:-1]
            ends = world_points[1:]
            colors = [(0, 1, 0, 1)] * len(starts)  # Green
            widths = [2.0] * len(starts)
            
            # Render the lines
            if self.debug_waypoints:
                self.draw.draw_lines(starts, ends, colors, widths)

            all_env_points_list.extend(world_points)

        self.lane_points_tensor = torch.stack([torch.stack(env_points) for env_points in all_env_points_list])
        self.lane_points_tensor = self.lane_points_tensor.view(self.num_envs, -1, 3)

        # In __init__
        self.previous_target_pos = None
        self.active_target_pos = None
        self.target_reached_threshold = 0.15
        self.current_waypoint_idx = 0

    def _calculate_cloeset_waypoint(self):
        if self.debug_waypoints:
            self.draw.clear_points()

        all_points_flat = self.lane_points_tensor.view(-1, 3).cpu().numpy().tolist()
        bg_colors = [[1.0, 1.0, 0.0, 0.3]] * len(all_points_flat) # Yellow (Faded)
        if self.debug_waypoints:
            self.draw.draw_points(all_points_flat, bg_colors, [15.0] * len(all_points_flat))

        # 1. Get current robot state
        robot_pos = self.robot.data.root_pos_w[:, :2]
        quat = self.robot.data.root_quat_w[:, :]
        
        # Calculate current heading vector (Forward is +X in standard yaw)
        current_yaw = torch.atan2(2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]), 
                                  1.0 - 2.0 * (quat[:, 2]**2 + quat[:, 3]**2))
        current_yaw += 3.14159  # Adjusting for your specific robot offset
        
        forward_vecs = torch.stack([torch.cos(current_yaw), torch.sin(current_yaw)], dim=-1)

        # 2. Filter for "Front" Waypoints
        # Calculate vectors from robot to ALL waypoints
        waypoint_vecs = self.lane_points_tensor[:, :, :2] - robot_pos[:, None, :] # [num_envs, num_waypoints, 2]

        dists_to_all = torch.norm(waypoint_vecs, dim=-1) # [num_envs, num_waypoints]
        
        # Use Dot Product to find which points are in front (> 0 means in front)
        # (N, 2) dot (2,) -> (N,)
        is_front = torch.sum(waypoint_vecs * forward_vecs[:, None, :], dim=-1) > 0 # [num_envs, num_waypoints]
        front_points_tensor = self.lane_points_tensor[is_front] # Result shape: [Total_Front_Points, 3]

        # 3. Move to CPU and convert to list for the debug drawer
        front_points_list = front_points_tensor.cpu().numpy().tolist()

        # 4. Define style
        num_front = len(front_points_list)
        if num_front > 0:
            # Bright Blue (RGBA) as per your color choice
            colors = [[0.0, 0.0, 1.0, 1.0]] * num_front 
            sizes = [15.0] * num_front

            if self.debug_waypoints:
                # 5. Render
                self.draw.draw_points(front_points_list, colors, sizes)

        if self.active_target_pos is not None:
            distances = torch.norm(self.active_target_pos - robot_pos, dim=-1) # [num_envs]
        else:
            # First frame initialization: force update for all
            distances = torch.zeros(self.num_envs, device=self.device)
            self.active_target_pos = robot_pos.clone()

        # 2. Identify which environments need a new target
        # Logic: Target is None OR we reached the current one (dist < 0.1)
        needs_new_target = (distances < 0.1)
        if torch.any(needs_new_target):
            # Calculate distances to ALL front waypoints
            front_dists = torch.norm(waypoint_vecs, dim=-1) # [num_envs, 109]

            # Mask "back" points by setting dist to infinity
            front_dists[~is_front] = float('inf')

            # Find closest front waypoint index for every env
            closest_front_idx = torch.argmin(front_dists, dim=-1) # [num_envs]

            # Add look-ahead (e.g., +3)
            target_indices = (closest_front_idx + 2) % self.lane_points_tensor.shape[1]

            # 3. Apply updates ONLY to envs that need them
            # We use advanced indexing to pull the correct points
            env_ids = torch.where(needs_new_target)[0]
            
            # Update active targets for specific environments
            new_found_targets = self.lane_points_tensor[env_ids, target_indices[env_ids], :2]
            self.active_target_pos[env_ids] = new_found_targets
        
        # Draw the visual markers for all active targets
        if self.active_target_pos is not None:
            # 1. Prepare Z-coordinates for all environments [num_envs, 1]
            z_offsets = torch.full((self.num_envs, 1), -0.2, device=self.device)
            
            # 2. Combine (X, Y) with Z -> [num_envs, 3]
            # This creates a tensor of [x, y, -0.2] for every environment
            targets_3d_tensor = torch.cat([self.active_target_pos, z_offsets], dim=-1)
            
            # 3. Convert to list for the debug drawer
            targets_3d_list = targets_3d_tensor.cpu().numpy().tolist()
            
            # 4. Define styles (one for each point)
            point_colors = [[1.0, 0.0, 0.0, 1.0]] * self.num_envs # Bright Red
            point_sizes = [20.0] * self.num_envs # Slightly larger to stand out
            
            if self.debug_waypoints:
                # 5. Render all target points at once
                self.draw.draw_points(targets_3d_list, point_colors, point_sizes)

        return distances

    def _move_robot_to_cloeset_waypoint(self):
        # --- 3. Calculate Steering for ALL Robots ---
        # Both are [num_envs, 2], so target_vec is [num_envs, 2]
        robot_pos = self.robot.data.root_pos_w[:, :2]
        quat = self.robot.data.root_quat_w[:, :]
        
        # Calculate current heading vector (Forward is +X in standard yaw)
        current_yaw = torch.atan2(2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]), 
                                  1.0 - 2.0 * (quat[:, 2]**2 + quat[:, 3]**2))
        current_yaw += 3.14159  # Adjusting for your specific robot offset

        target_vec = self.active_target_pos - robot_pos

        # Calculate target yaw for each robot
        # target_vec[:, 1] is Y, target_vec[:, 0] is X
        target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])

        # yaw_error will be [num_envs]
        yaw_error = target_yaw - current_yaw

        # Normalize error to [-pi, pi] for all environments
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))

        # --- 4. Control Output (Batch Processed) ---
        base_vel = 0.6
        steer_gain = 4.0

        # Calculate speed factor for each robot based on its own yaw error
        speed_factor = torch.clamp(1.0 - torch.abs(yaw_error) / 1.5, min=0.3)

        # Calculate wheel velocities for all robots [num_envs]
        left_wheel = (base_vel * speed_factor) - (yaw_error * steer_gain)
        right_wheel = (base_vel * speed_factor) + (yaw_error * steer_gain)

        # --- 5. Apply Actions ---
        # Stack the wheels to create a [num_envs, 2] tensor
        # We use dim=-1 to join them as columns
        actions = torch.stack([left_wheel, right_wheel], dim=-1) * 10.0

        # Apply the vectorized actions to the simulation
        actions = torch.tensor([[0.0, 0.0]], device='cuda:0')
        self.robot.set_joint_velocity_target(actions, joint_ids=self.dof_idx)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        if self.active_target_pos != None:
            self._move_robot_to_cloeset_waypoint()
        else:
           self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)
        
    def _get_observations(self) -> dict:
        self.velocity = self.robot.data.root_com_vel_w 
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)

        forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)

        obs = forward_speed

        left_camera_image = self.scene["left_camera"].data.output["rgb"]
        right_camera_image = self.scene["right_camera"].data.output["rgb"]
        def expand_scalar_to_image(scalar_obs, image_shape):
            # scalar_obs shape: [B, S] (e.g., [2, 2])
            # image_shape: (B, C, H, W) (e.g., [2, 3, 480, 480])
            B, S = scalar_obs.shape
            _, _, H, W = image_shape

            # 1. Reshape scalar to [B, S, 1, 1]
            scalar_expanded = scalar_obs.view(B, S, 1, 1)

            # 2. Tile/Expand to [B, S, H, W]
            # This creates a "channel" where every pixel is the same scalar value
            scalar_map = scalar_expanded.expand(-1, -1, H, W)

            return scalar_map

        # obs.shape:  torch.Size([2, 2])
        left_camera_input = left_camera_image.permute(0, 3, 1, 2).float() / 255.0
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=left_camera_input.device).view(1, 3, 1, 1)
        left_camera_gray = (left_camera_input * weights).sum(dim=1, keepdim=True)
        left_camera_gray = F.interpolate(left_camera_gray, size=(64, 64), mode='bilinear', align_corners=False)

        right_camera_input = right_camera_image.permute(0, 3, 1, 2).float()  / 255.0
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=right_camera_input.device).view(1, 3, 1, 1)
        right_camera_gray = (right_camera_input * weights).sum(dim=1, keepdim=True)
        right_camera_gray = F.interpolate(right_camera_gray, size=(64, 64), mode='bilinear', align_corners=False)
        # left_camera_input.shape:  torch.Size([2, 3, 480, 640])

        # 2. Expand scalar_obs to match image spatial dimensions [480, 640]
        B, S = obs.shape
        H, W = left_camera_gray.shape[2], left_camera_gray.shape[3]

        # Reshape [2, 2] -> [2, 2, 1, 1] then tile to [2, 2, 480, 640]
        scalar_map = obs.view(B, S, 1, 1).expand(-1, -1, H, W)

        # left_camera_input.shape:  torch.Size([2, 3, 480, 640])
        # scalar_map.shape:  torch.Size([2, 2, 480, 640])
        combined_input = torch.cat([left_camera_gray, scalar_map], dim=1)
        combined_input = torch.cat([combined_input, right_camera_gray], dim=1)

        observations = {"policy": combined_input}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        distances = self._calculate_cloeset_waypoint()

        #self.draw.clear_points()
        '''
        if self.active_target_pos is not None:
            z_offsets = torch.full((self.num_envs, 1), -0.2, device=self.device)
            targets_3d_tensor = torch.cat([self.active_target_pos, z_offsets], dim=-1)
            targets_3d_list = targets_3d_tensor.cpu().numpy().tolist()
            point_colors = [[1.0, 0.0, 0.0, 1.0]] * self.num_envs
            point_sizes = [20.0] * self.num_envs
            self.draw.draw_points(targets_3d_list, point_colors, point_sizes)
        '''

        #total_reward = needs_new_target.float().unsqueeze(1)
        distance_reward = -distances
        distance_reward = distance_reward.float().unsqueeze(1)
        distance_reward = torch.clamp(distance_reward, min=-10.0)
        total_reward = distance_reward

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)

        self.previous_target_pos = None
        self.active_target_pos = None
        self.target_reached_threshold = 0.15
        self.current_waypoint_idx = 0

        num_resets = len(env_ids)
        
        # 1. Get the number of waypoints per track (the second dimension)
        # self.lane_points_tensor shape is [num_envs, 109, 3]
        num_waypoints_per_env = self.lane_points_tensor.shape[1]
        
        # 2. Pick a random waypoint index for each resetting environment
        random_waypoint_indices = torch.randint(0, num_waypoints_per_env, (num_resets,), device=self.device)
        #print("random_waypoint_indices: ", random_waypoint_indices)

        # 3. Advanced Indexing: 
        # We need: self.lane_points_tensor[env_id, random_waypoint_index]
        # This ensures Env 0 stays on Track 0, etc.
        spawn_pos = self.lane_points_tensor[env_ids, random_waypoint_indices].clone()
        
        # Lift slightly above floor
        spawn_pos[:, 2] += 0.2
        
        # 4. Create the orientation (Shape: [num_resets, 4])
        base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        spawn_quat = base_quat.unsqueeze(0).repeat(num_resets, 1)
        
        # 5. Concatenate to [num_resets, 7]
        spawn_pose = torch.cat([spawn_pos, spawn_quat], dim=-1)
        
        # 6. Apply to simulation
        self.robot.write_root_pose_to_sim(spawn_pose, env_ids)
        
        # 7. Reset velocities to zero
        zeros = torch.zeros((num_resets, 6), device=self.device)
        self.robot.write_root_velocity_to_sim(zeros, env_ids)
        
        self._calculate_cloeset_waypoint()

        target_pos = self.active_target_pos[env_ids]

        # Extract relevant coordinates
        target_x = target_pos[:, 0]
        target_y = target_pos[:, 1]
        spawn_x = spawn_pos[:, 0]
        spawn_y = spawn_pos[:, 1]

        # 1. Calculate the relative displacement
        dx = target_x - spawn_x
        dy = target_y - spawn_y

        # 2. Compute Yaw (Angle around Z-axis)
        # atan2 handles the quadrant logic automatically
        yaw = torch.atan2(dy, dx) + torch.pi

        # 3. Create Quaternions (w, x, y, z)
        # For a robot rotating only on the ground (Z-axis):
        # w = cos(yaw/2), x = 0, y = 0, z = sin(yaw/2)
        num_resets = yaw.shape[0]
        spawn_quat = torch.zeros((num_resets, 4), device='cuda:0')
        spawn_quat[:, 0] = torch.cos(yaw / 2.0)  # w (scalar part)
        spawn_quat[:, 3] = torch.sin(yaw / 2.0)  # z (vector part)

        # 4. Final Pose [num_resets, 7]
        spawn_pose = torch.cat([spawn_pos, spawn_quat], dim=-1)
        self.robot.write_root_pose_to_sim(spawn_pose, env_ids)
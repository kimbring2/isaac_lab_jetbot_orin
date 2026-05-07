# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import itertools
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
#from isaaclab.utils import set_seed

from isaaclab.sensors import Camera, CameraCfg, TiledCamera
from isaaclab.assets import AssetBase, AssetBaseCfg

from isaacsim.core.utils.extensions import enable_extension
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.xforms import get_world_pose

#import isaacsim.util.debug_draw._debug_draw as _debug_draw

def quat_to_euler(q):
    """
    Converts quaternions [x, y, z, w] to Euler angles [roll, pitch, yaw]
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, 
                        torch.sign(sinp) * (torch.pi / 2), 
                        torch.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def euler_to_quat(euler):
    """
    Converts Euler angles [roll, pitch, yaw] to quaternions [x, y, z, w]
    """
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy

    return torch.stack([x, y, z, w], dim=-1)


class IsaacLabJetbotOrinEnv(DirectRLEnv):
    cfg: IsaacLabJetbotOrinEnvCfg

    def __init__(self, cfg: IsaacLabJetbotOrinEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

        #self.seed(42)

    def _setup_scene(self):
        # For draw debugging
        #self.draw = _debug_draw.acquire_debug_draw_interface()

        # 1. Spawn the track first (it's the floor)
        self.track = RigidObject(self.cfg.track_cfg)
        self.scene.rigid_objects["track"] = self.track

        # 2. Spawn the robot
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # 3. clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # 4. add left camera
        self.left_camera = TiledCamera(self.cfg.left_tiled_camera)
        self.scene.sensors["left_camera"] = self.left_camera

        # 5. add right camera
        self.right_camera = TiledCamera(self.cfg.right_tiled_camera)
        self.scene.sensors["right_camera"] = self.right_camera

        # 6. add lights
        # Example per‑env light (if you want per-env control)
        light_cfg_1 = sim_utils.SphereLightCfg(intensity=2000.0, radius=1.0, color=(0.75, 0.75, 0.75), )
        light_path = f"/World/envs/env_0/Light1"
        light_cfg_1.func(light_path, light_cfg_1, translation=(0.0, 0.0, 1.0))

        light_cfg_2 = sim_utils.SphereLightCfg(intensity=2000.0, radius=1.0, color=(0.75, 0.75, 0.75),)
        light_path = f"/World/envs/env_0/Light2"
        light_cfg_2.func(light_path, light_cfg_2, translation=(0.0, -1.0, 1.0))

        light_cfg_3 = sim_utils.SphereLightCfg(intensity=2000.0, radius=1.0, color=(0.75, 0.75, 0.75),)
        light_path = f"/World/envs/env_0/Light3"
        light_cfg_3.func(light_path, light_cfg_3, translation=(-1.5, 0.0, 1.0))

        light_cfg_4 = sim_utils.SphereLightCfg(intensity=2000.0, radius=1.0, color=(0.75, 0.75, 0.75),)
        light_path = f"/World/envs/env_0/Light4"
        light_cfg_4.func(light_path, light_cfg_4, translation=(-1.5, -1.0, 1.0))

        # 7. add spawn point
        parent_prim = prim_utils.get_prim_at_path("/World/envs/env_0/Track/Spawn_Point")
        spawn_point_prims = prim_utils.get_prim_children(parent_prim)

        # Extract local points once (Template for env_0)
        # We use a list comprehension to get the poses from the Prims
        poses = [get_world_pose(p.GetPath().pathString) for p in spawn_point_prims]

        # Convert to tensors and make relative to env_0
        # [17, 3] and [17, 4]
        local_pos = torch.tensor([p[0] for p in poses], device=self.device) - self.scene.env_origins[0]

        self.spawn_quat_tensor = torch.tensor([p[1] for p in poses], device=self.device)
        angle_rad = torch.tensor([np.pi / 2], device=self.device)
        zeros = torch.zeros_like(angle_rad)
        rotation_offset = math_utils.quat_from_euler_xyz(zeros, zeros, angle_rad)
        rotation_repeat = rotation_offset.repeat(self.spawn_quat_tensor.shape[0], 1)
        self.spawn_quat_tensor = math_utils.quat_mul(self.spawn_quat_tensor, rotation_repeat)

        # Vectorized Calculation for ALL environments
        # self.scene.env_origins shape: [num_envs, 3]
        # local_pos shape: [17, 3]
        # Resulting global_spawn_pos shape: [num_envs, 17, 3]
        self.spawn_pos_tensor = (local_pos[None, :]) + self.scene.env_origins[:, None]
        self.spawn_pos_tensor[:, :, 2] += 0.0  # Apply Z-offset

        all_spawn_points = self.spawn_pos_tensor.reshape(-1, 3).tolist()

        #self.draw.draw_points(all_spawn_points, [[0, 1, 0, 1]] * len(all_spawn_points), [5.0] * len(all_spawn_points))

        # Define a local 'forward' vector (0.1 meters long in the X direction)
        # We repeat it for every single spawn point in the scene
        # Calculate total number of points across all envs
        num_envs = self.num_envs
        num_points_per_env = self.spawn_quat_tensor.shape[0] # Should be 17
        total_points = num_envs * num_points_per_env        # e.g., 289

        # Prepare local_forward [Total_Points, 3]
        local_forward = torch.tensor([0.1, 0.0, 0.0], device=self.device, dtype=torch.float32)
        local_forward = local_forward.repeat(total_points, 1)

        # Prepare all_quats [Total_Points, 4]
        # We repeat the 17 template quats for every environment
        all_quats = self.spawn_quat_tensor.repeat(num_envs, 1).to(dtype=torch.float32)

        # Apply rotation - Now both are (Total_Points, ...)
        world_forward_vecs = math_utils.quat_apply(all_quats, local_forward)

        # Calculate start/end and draw
        line_starts = self.spawn_pos_tensor.reshape(-1, 3)
        line_ends = line_starts + world_forward_vecs

        '''
        self.draw.draw_lines(
            line_starts.tolist(), line_ends.tolist(), [[1.0, 0.0, 0.0, 1.0]] * total_points,  [2.0] * total_points
        )
        '''

        # 8. add road lane
        # Get the parent scope prim
        parent_path = "/World/envs/env_0/Track/Road_Lane"
        parent_prim = prim_utils.get_prim_at_path(parent_path)

        # Iterate through children and extract points
        all_lane_segments_a = []
        all_lane_segments_b = []

        for child in prim_utils.get_prim_children(parent_prim):
            if child.GetTypeName() == "BasisCurves":
                curve_geom = UsdGeom.BasisCurves.Get(self.sim.stage, child.GetPath())
                local_pts = torch.tensor(curve_geom.GetPointsAttr().Get(), device=self.device)
                
                # Scale and offset to world coordinates for the template (env_0)
                world_pts = (local_pts * 0.05) + self.scene.env_origins[0]
                
                # Create segments (Start A to End B)
                #print("world_pts[:-1, :2]: ", world_pts[:-1, :2])
                all_lane_segments_a.append(world_pts[:-1, :2])
                all_lane_segments_b.append(world_pts[1:, :2])

        # Combine into master tensors and broadcast to all environments
        # Shape: [num_total_segments, 2] -> [num_envs, num_total_segments, 2]
        self.master_seg_a = torch.cat(all_lane_segments_a, dim=0).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.master_seg_b = torch.cat(all_lane_segments_b, dim=0).unsqueeze(0).repeat(self.num_envs, 1, 1)

        # Shift env_0 template to each environment's specific origin
        # Since template was env_0, we subtract origin[0] and add all origins
        origin_offsets = self.scene.env_origins - self.scene.env_origins[0]
        self.master_seg_a += origin_offsets[:, None, :2]
        self.master_seg_b += origin_offsets[:, None, :2]

        # Concatenate segments (assuming they are [N, 2] from your previous loop)
        template_starts_2d = torch.cat(all_lane_segments_a, dim=0)
        template_ends_2d = torch.cat(all_lane_segments_b, dim=0)

        # Add a Z-axis column of zeros to make them [N, 3]
        # This prevents the "size 2 must match size 3" error
        template_starts = torch.cat([template_starts_2d, torch.zeros((template_starts_2d.shape[0], 1), device=self.device)], dim=-1)
        template_ends = torch.cat([template_ends_2d, torch.zeros((template_ends_2d.shape[0], 1), device=self.device)], dim=-1)

        # Broadcast to all environments
        # origin_offsets shape: [num_envs, 3]
        origin_offsets = self.scene.env_origins - self.scene.env_origins[0]

        # Final Shapes: [num_envs, Total_Segments, 3]
        # Addition now works because both tensors have 3 as the last dimension
        all_envs_starts = template_starts.unsqueeze(0) + origin_offsets.unsqueeze(1)
        all_envs_ends = template_ends.unsqueeze(0) + origin_offsets.unsqueeze(1)

        # Add a tiny Z-offset for visualization ONLY to prevent "flickering"
        # This makes the lines float 2cm above the ground
        all_envs_starts[:, :, 2] += -0.19
        all_envs_ends[:, :, 2] += -0.19

        # Flatten to lists for the Isaac Lab debug drawer
        line_starts_list = all_envs_starts.reshape(-1, 3).tolist()
        line_ends_list = all_envs_ends.reshape(-1, 3).tolist()

        # Render the lines
        #colors = [[1.0, 0.5, 0.0, 1.0]] * len(line_starts_list) # Orange
        #widths = [2.0] * len(line_starts_list)
        #self.draw.draw_lines(line_starts_list, line_ends_list, colors, widths)

        self.milestones_reached = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.bool)
        self.total_step = 0
        self.total_reward = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

        #self.frame_buffer = torch.zeros(
        #    (self.num_envs, 4, 3, 64, 64), device=self.device
        #)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        '''
        if self.active_target_pos != None:
            actions, yaw_error = self._move_robot_to_cloeset_waypoint()
        else:
            self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)
            actions = self.actions
        
        self.actions = actions
        '''

        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        robot_pos = self.robot.data.root_pos_w
        robot_z_position = robot_pos[:, 2]

        robot_fell_off = robot_z_position < -0.2
        fell_off_env_ids = robot_fell_off.nonzero(as_tuple=False).flatten()

        # Call if robot_fell_off is True _reset_idx(self, env_ids: Sequence[int] | None):
        if len(fell_off_env_ids) > 0:
            self._reset_idx(fell_off_env_ids)

        self.velocity = self.robot.data.root_com_vel_w 
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)

        forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        obs = -forward_speed

        left_camera_image = self.scene["left_camera"].data.output["rgb"]
        right_camera_image = self.scene["right_camera"].data.output["rgb"]

        # obs.shape:  torch.Size([2, 2])
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=left_camera_image.device).view(1, 3, 1, 1)

        left_camera_input = left_camera_image.permute(0, 3, 1, 2).float() / 255.0
        left_camera_gray = (left_camera_input * weights).sum(dim=1, keepdim=True)
        left_camera_gray = F.interpolate(left_camera_gray, size=(64, 64), mode='bilinear', align_corners=False)

        right_camera_input = right_camera_image.permute(0, 3, 1, 2).float() / 255.0
        right_camera_gray = (right_camera_input * weights).sum(dim=1, keepdim=True)
        right_camera_gray = F.interpolate(right_camera_gray, size=(64, 64), mode='bilinear', align_corners=False)



        # Define noise intensity (0.02 is a good starting point for moderate noise)
        cam_noise_std = 0.02 

        # Add Gaussian noise to left camera
        left_noise = torch.randn_like(left_camera_gray) * cam_noise_std
        left_camera_gray = torch.clamp(left_camera_gray + left_noise, 0.0, 1.0)

        # Add Gaussian noise to right camera
        right_noise = torch.randn_like(right_camera_gray) * cam_noise_std
        right_camera_gray = torch.clamp(right_camera_gray + right_noise, 0.0, 1.0)

        #mask = torch.rand_like(left_camera_gray) > 0.01
        #left_camera_gray *= mask
        #right_camera_gray *= mask

        # Expand scalar_obs to match image spatial dimensions [480, 640]
        B, S = obs.shape
        H, W = left_camera_gray.shape[2], left_camera_gray.shape[3]

        # Reshape [2, 2] -> [2, 2, 1, 1] then tile to [2, 2, 480, 640]
        scalar_map = obs.view(B, S, 1, 1).expand(-1, -1, H, W) / 2.0

        combined_input = torch.cat([left_camera_gray, scalar_map], dim=1)
        combined_input = torch.cat([combined_input, right_camera_gray], dim=1)

        observations = {"policy": combined_input}

        self.total_step += 1

        return observations

    def _get_rewards(self) -> torch.Tensor:
        #self.draw.clear_lines()

        robot_pos_world = self.robot.data.root_pos_w
        robot_pos_local = robot_pos_world - self.scene.env_origins
        robot_x_position = robot_pos_local[:, 0]

        #print("robot_x_position.shape: ", robot_x_position.shape)
        reach_goal = robot_x_position >= 0.85
        reach_goal_env_ids = reach_goal.nonzero(as_tuple=False).flatten()

        total_reward = torch.zeros((self.num_envs, 1), device=self.device)

        # Define the 4 levels (milestones)
        levels = torch.tensor([-0.35, 0.05, 0.45, 0.85], device=self.device)
        level_rewards = torch.tensor([0.25, 0.50, 0.75, 1.0], device=self.device)

        # Check each level
        for i in range(len(levels)):
            # Find envs that are past the level AND haven't received the reward yet
            passed_level = robot_x_position >= levels[i]
            new_milestone = passed_level & ~self.milestones_reached[:, i]
            
            # Give the reward only to those environments
            #total_reward[new_milestone, 0] += level_rewards[i]
            
            # Mark as reached so they don't get it again next frame
            self.milestones_reached[new_milestone, i] = True
        
        #print("self.milestones_reached: ", self.milestones_reached)

        # Call if robot_fell_off is True _reset_idx(self, env_ids: Sequence[int] | None):
        if len(reach_goal_env_ids) > 0:
            #print("reach_goal_env_ids: ", reach_goal_env_ids)
            #total_reward[reach_goal_env_ids, 0] += 1.0
            self._reset_idx(reach_goal_env_ids)

        forward_reward = -self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        #print("forward_reward: ", forward_reward / 10.0)
        total_reward += forward_reward / 10.0

        # Inside _get_rewards
        robot_pos = self.robot.data.root_pos_w[:, :2] # [num_envs, 2]

        # Create the Proximity Mask [num_envs, num_segs]
        # Only consider segments within 2.0 meters of each robot
        dist_to_starts = torch.norm(self.master_seg_a - robot_pos[:, None, :], dim=-1)
        proximity_mask = dist_to_starts < 2.0

        # Mask the Segment Tensors
        # We use the mask to keep segments near the robot and replace far ones with 0
        # We unsqueeze(-1) to broadcast the [2, 500] mask to the [2, 500, 2] coordinates
        masked_seg_a = torch.where(proximity_mask.unsqueeze(-1), self.master_seg_a, torch.zeros_like(self.master_seg_a))
        masked_seg_b = torch.where(proximity_mask.unsqueeze(-1), self.master_seg_b, torch.zeros_like(self.master_seg_b))

        ## --- Lane Detection Logic ---
        # 1. Get current robot position [num_envs, 2]
        robot_xy = self.robot.data.root_pos_w[:, :2]

        # 2. Define vectors for segments and robot relative to segment starts
        # line_vecs: [num_envs, num_segs, 2]
        # robot_vecs: [num_envs, num_segs, 2]
        line_vecs = masked_seg_b - masked_seg_a
        robot_vecs = robot_xy[:, None, :] - masked_seg_a

        # 3. Calculate squared length of segments for normalization
        seg_len_sq = torch.sum(line_vecs**2, dim=-1) + 1e-6

        # 4. Find the projection factor 't' 
        # t=0 at start point A, t=1 at end point B
        t = torch.sum(robot_vecs * line_vecs, dim=-1) / seg_len_sq

        # 5. Calculate Perpendicular (Lateral) Distance
        # Cross product: (Ax * By) - (Ay * Bx)
        cross_prod = (line_vecs[..., 0] * robot_vecs[..., 1]) - (line_vecs[..., 1] * robot_vecs[..., 0])
        line_lengths = torch.sqrt(seg_len_sq)
        perp_dists = torch.abs(cross_prod) / line_lengths

        # 6. IDENTIFY IF ROBOT IS "ON" THE LINE
        # Condition A: Robot projection is between the start and end of the segment (0 < t < 1)
        # Condition B: Robot is physically close enough to the line (e.g., within 0.1m)
        is_within_bounds = (t >= 0.0) & (t <= 1.0)
        is_close_perpendicularly = perp_dists < 0.025  # Adjust this threshold to your robot's width

        # Final Mask: [num_envs, num_segs] 
        # This is True for every segment the robot is currently overlapping
        robot_is_on_segment = is_within_bounds & is_close_perpendicularly

        ## 7. VISUALIZATION (Per-Environment)
        # Loop through each environment
        for env_idx in range(self.num_envs):
            # Get the mask for ONLY this specific environment [138]
            env_mask = robot_is_on_segment[env_idx]
            
            # Check if this specific robot is on any segment
            if torch.any(env_mask):
                # Extract segments for this environment only
                # Resulting shape: [num_hit_segments, 2]
                on_starts = masked_seg_a[env_idx][env_mask]
                on_ends = masked_seg_b[env_idx][env_mask]

                if sum(sum(on_starts - on_ends)).any():
                    # Prepare drawing coordinates
                    # Call if robot_fell_off is True _reset_idx(self, env_ids: Sequence[int] | None):
                    total_reward[[env_idx], :] -= 1.0
                    self._reset_idx(torch.tensor([env_idx], device=self.device))

                    z_height = torch.full((on_starts.shape[0], 1), -0.18, device=self.device)
                    draw_starts = torch.cat([on_starts, z_height], dim=-1).tolist()
                    draw_ends = torch.cat([on_ends, z_height], dim=-1).tolist()
                    
                    # Pick a color based on the environment index
                    # Env 0 = Red, Env 1 = Green, Others = Yellow
                    color = [1.0, 0.0, 0.0, 1.0] # Red
                    colors = [color] * len(draw_starts)
                    
                    # Draw lines for this specific environment
                    #self.draw.draw_lines(draw_starts, draw_ends, colors, [10.0] * len(draw_starts))
        
        self.total_reward += total_reward
        reset_mask = self.total_reward.squeeze(-1) > 5.0

        # 2. Get the indices of those environments
        env_ids = reset_mask.nonzero(as_tuple=False).flatten()

        # 3. If any environments need resetting:
        if len(env_ids) > 0:
            # Reset the reward tracker for these specific envs
            self.total_reward[env_ids] = 0.0
            
            # Call your reset function with the batch of IDs
            self._reset_idx(env_ids)

        #print("self.total_reward: ", self.total_reward)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        self.reset_buf[env_ids] = False
        self.milestones_reached[env_ids] = 0

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        
        # 2. Reset the actions buffer to zero
        # This ensures no old commands persist in the next step
        self.actions[env_ids] = 0.0
        self.total_reward[env_ids] = 0.0

        # 1. Determine how many spawn points are available per environment (usually 17)
        num_points_per_env = self.spawn_pos_tensor.shape[1]

        # 2. Pick a random spawn point index for each resetting environment
        # Shape: [num_resets]
        point_indices = torch.randint(0, num_points_per_env, (len(env_ids),), device=self.device)

        # 3. Select the specific positions and rotations
        # Use advanced indexing: self.spawn_pos_tensor[env_ids, point_indices]
        # Shape: [num_resets, 3] and [num_resets, 4]
        selected_pos = self.spawn_pos_tensor[env_ids, point_indices]
        selected_quat = self.spawn_quat_tensor[point_indices] # Rotation is usually local/same for all envs
        #selected_quat = torch.tensor([[1.0000e+00,  2.1049e-08,  2.1049e-08,  1.1555e-03]], device='cuda:0', dtype=torch.float64)
        #print("selected_quat: ", selected_quat)
        # selected_quat:  tensor([[-1.0000e+00,  2.1049e-08,  2.1049e-08,  1.1555e-03]], device='cuda:0', dtype=torch.float64)

        # 1. Convert the current orientation to Euler angles (Roll, Pitch, Yaw)
        euler_angles = quat_to_euler(selected_quat) # Shape: [num_resets, 3]

        # 2. Create a random mask (50% chance to flip)
        flip_mask = torch.rand(len(env_ids), device=self.device) > 0.5

        # 3. Add 180 degrees (pi) to the Yaw (index 2) where the mask is True
        # We use torch.pi for 180 degrees
        euler_angles[flip_mask, 0] += torch.pi

        # 4. Convert back to Quaternion
        selected_quat_rotated = euler_to_quat(euler_angles)

        # 4. Prepare the new root state
        # Root state format: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        # We clone the default state to ensure velocities are zeroed out
        root_state = self.robot.data.default_root_state[env_ids].clone()

        # Update Position and Orientation
        root_state[:, :3] = selected_pos
        root_state[:, 2] += 0.15
        root_state[:, 3:7] = selected_quat

        # 5. Write the state back to the physics simulation
        self.robot.write_root_state_to_sim(root_state, env_ids)

        # Set the seed at the very beginning of your main function
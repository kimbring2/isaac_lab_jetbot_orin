# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import os

from isaac_lab_jetbot_orin.robots.jetbot import JETBOT_CONFIG

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaac_lab_jetbot_orin.randomization.light_dr import randomize_lights
from isaac_lab_jetbot_orin.randomization.mesh_dr import change_track_texture, change_curtain_texture
from isaac_lab_jetbot_orin.randomization.camera_dr import randomize_camera_parameters
from isaac_lab_jetbot_orin.randomization.motor_dr import randomize_motor_parameters


@configclass
class JetbotEventCfg:
    # Randomize lights at every episode reset
    randomize_env_lights = EventTerm(
        func=randomize_lights,
        mode="reset",
        params={
            "intensity_range": (500.0, 4000.0),
            "color_range": ((0.4, 0.4, 0.4), (1.0, 1.0, 1.0)),
        },
    )

    startup_track_randomization = EventTerm(
        func=change_track_texture,
        mode="startup",
        params={
            "texture_folder_path": "source/isaac_lab_jetbot_orin/assets/Collected_starter_kit_track/textures",
        },
    )
    
    randomize_track_appearance = EventTerm(
        func=change_track_texture,
        mode="interval",
        interval_range_s=(300, 300),
        params={
            "texture_folder_path": "source/isaac_lab_jetbot_orin/assets/Collected_starter_kit_track/textures",
        },
    )
    
    randomize_curtain_appearance = EventTerm(
        func=change_curtain_texture,
        mode="interval", # Change from 'reset' to 'interval'
        interval_range_s=(30, 30), # Fires exactly every 5000 steps
    )

    randomize_camera_parameters = EventTerm(
        func=randomize_camera_parameters,
        mode="interval",
        interval_range_s=(30, 30),
        params={
            "focal_length_range": (2.3, 2.9),
            "focus_dist_range": (55.0, 66.0)
        },
    )

    randomize_motor_parameters = EventTerm(
        func=randomize_motor_parameters,
        mode="interval",
        interval_range_s=(30, 30),
        params={
            "damping_range": (160.0, 190.0),
            "stiffness_range": (0.0, 5.0),
        },
    )

 
@configclass
class IsaacLabJetbotOrinEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 8
    episode_length_s = 30.0
    
    # - spaces definition
    action_space = spaces.Box(low=-7.5, high=7.5, shape=(2,), dtype=np.float32)
    #action_space = gym.spaces.Discrete(25)
    
    #observation_space = 3
    observation_space = spaces.Box(
        low=-0.0, 
        high=1.0, 
        shape=(3, 64, 64), 
        dtype=np.float32
    )

    state_space = 0
    
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    
    # Track Configuration
    # We use AssetBaseCfg for a static mesh that doesn't need joint control
    #usd_path = os.path.join(os.getcwd(), "source/isaac_lab_jetbot_orin/assets/Collected_starter_kit_track/simple_straight_track.usd")
    usd_path = os.path.join(os.getcwd(), "source/isaac_lab_jetbot_orin/assets/Collected_starter_kit_track/simple_curved_track.usd")
    #usd_path = os.path.join(os.getcwd(), "source/isaac_lab_jetbot_orin/assets/Collected_starter_kit_track/starter_kit_track.usd")

    track_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Track",
        spawn=sim_utils.UsdFileCfg(
            #usd_path="/media/kimbring2/be356a87-def6-4be8-bad2-077951f0f3da/isaac_lab_jetbot_orin/source/isaac_lab_jetbot_orin/assets/Collected_starter_kit_track/starter_kit_track.usd",
            usd_path=usd_path,
            scale=(0.05, 0.05, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, 
                disable_gravity=True
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.2)),
    )

    left_tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/Body/tiled_camera_0",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.07617, -0.02921, -0.03784), 
                                        rot=(0.5792, 0.40558, 0.40558, 0.57923), 
                                        convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.6, focus_distance=60.0, horizontal_aperture=3.67, clipping_range=(0.01, 5.0)
        ),
        width=640,
        height=480,
    )

    right_tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/Body/tiled_camera_1",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.07617, 0.03021, -0.03784), 
                                        rot=(0.5792, 0.40558, 0.40558, 0.57923), 
                                        convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.6, focus_distance=60.0, horizontal_aperture=3.67, clipping_range=(0.01, 5.0)
        ),
        width=640,
        height=480,
    )

    # /home/kimbring2/isaac_lab_jetbot_orin/source/isaac_lab_jetbot_orin/assets/Collected_starter_kit_track/textures/straight/straight_1.png
    events: JetbotEventCfg = JetbotEventCfg()
    
    # robot(s)
    robot_cfg: ArticulationCfg = JETBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # scene
    #scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=2.0, replicate_physics=True)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100, 
        env_spacing=7.5, 
        replicate_physics=True,
    )

    dof_names = ["left_wheel_joint", "right_wheel_joint"]
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from gymnasium import spaces
import numpy as np
import os

from isaac_lab_jetbot_orin.robots.jetbot import JETBOT_CONFIG

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
#print("os.getcwd(): ", os.getcwd())
# os.getcwd():  /media/kimbring2/be356a87-def6-4be8-bad2-077951f0f3da/isaac_lab_jetbot_orin

@configclass
class IsaacLabJetbotOrinEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 50.0
    
    # - spaces definition
    action_space = 2
    
    #observation_space = 3
    observation_space = spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(7, 64, 64), 
        dtype=np.float32
    )

    state_space = 0
    
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    
    # Track Configuration
    # We use AssetBaseCfg for a static mesh that doesn't need joint control

    usd_path = os.path.join(os.getcwd(), "source/isaac_lab_jetbot_orin/assets/Collected_starter_kit_track/starter_kit_track.usd")
    track_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Track",
        spawn=sim_utils.UsdFileCfg(
            #usd_path="/media/kimbring2/be356a87-def6-4be8-bad2-077951f0f3da/isaac_lab_jetbot_orin/source/isaac_lab_jetbot_orin/assets/Collected_starter_kit_track/starter_kit_track.usd",
            usd_path=usd_path,
            scale=(0.1, 0.1, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, 
                disable_gravity=True
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.2)),
    )
    
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
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test Isaac Lab environment without skrl.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import torch
import time
import numpy as np

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
#import isaaclab_tasks  # noqa: F401
import isaac_lab_jetbot_orin.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

def main():
    # 1. Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )

    # 2. Create Isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 3. Optional: Convert multi-agent to single-agent if the task requires it
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Get environment dt for real-time sync
    dt = getattr(env, "step_dt", 0.01)

    print(f"[INFO] Environment setup complete. Action space: {env.action_space}")
    print(f"[INFO] Observation space: {env.observation_space}")

    # 4. Reset environment
    obs, _ = env.reset()

    # 5. Simulation loop
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # Generate random actions for all vectorized environments
            # env.action_space.sample() works with Gymnasium vectorized envs
            actions = env.action_space.sample()
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(device=args_cli.device)
            
            # Ensure actions is a 2D tensor (num_envs, action_dim)
            if actions.ndim == 1:
                actions = actions.unsqueeze(0)
            
            # Step the environment
            # actions.shape:  torch.Size([2, 2])
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Check if any environment needs a reset (IsaacLab usually handles auto-reset)
            if torch.any(terminated) or torch.any(truncated):
                # In vectorized envs, reset is often automatic, 
                # but you can handle manual logic here if needed.
                pass

        # Time delay for real-time evaluation
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Close the simulator
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
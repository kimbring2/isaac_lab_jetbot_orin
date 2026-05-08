# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher
import cv2
import torch.nn.functional as F
from torchvision.transforms import functional as VF

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import os
import time
import torch
import numpy as np

import skrl
from packaging import version

# Debug purpose - For keyboard manual control

from pynput import keyboard

key_input = 'f'

def on_press(key):
    global key_input
    try:
        # For alphanumeric keys (a-z, 0-9)
        key_input = key.char
    except AttributeError:
        # For special keys (Space, Arrows, etc.)
        # key.char doesn't exist here, so we use str(key) or a specific name
        key_input = str(key) 
        # Example: if you press Space, key_input will be 'Key.space'


def on_release(key):
    global key_input
    #print('{0} released'.format(key))
    key_input = 'f'
    if key == keyboard.Key.esc:
        # Stop listener
        return False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import isaac_lab_jetbot_orin.tasks  # noqa: F401
import isaaclab.sim as sim_utils

# config shortcuts
algorithm = args_cli.algorithm.lower()


##########################
import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space
from gymnasium import spaces
import gymnasium as gym
import numpy as np

action_space = spaces.Box(low=-7.5, high=7.5, shape=(2,), dtype=np.float32)
observation_space = spaces.Box(low=-0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32)

class GaussianModel(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-2, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        # Note: We use Lazy layers, so we must do a dummy pass before loading weights
        self.net_container = nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=5, stride=2),
            nn.ELU(),
            nn.LazyConv2d(out_channels=128, kernel_size=3, stride=2),
            nn.ELU(),
            nn.LazyConv2d(out_channels=128, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten(),
            nn.LazyLinear(out_features=1024),
            nn.ELU(),
            nn.LazyLinear(out_features=1024),
            nn.ELU(),
            nn.LazyLinear(out_features=self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)

    def compute(self, inputs, role=""):
        # The unflatten utility handles the conversion from a flat tensor to 
        # the (C, H, W) shape expected by your Conv2d layers.
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        output = self.net_container(states)
        
        return output, self.log_std_parameter, {}


# 2. Instantiate and Load Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Replace obs_space and act_space with your environment's actual dimensions
policy = GaussianModel(observation_space, action_space, device)

# Load the checkpoint6
checkpoint = torch.load('/home/kimbring2/isaac_lab_jetbot_orin/logs/skrl/jetbot_orin_direct/2026-05-08_05-17-11_ppo_torch/checkpoints/agent_5000.pt', map_location=device)

policy.load_state_dict(checkpoint['policy']) # skrl stores it under 'policy'
policy.eval()

dummy_input = {"states": torch.zeros((1, 3, 64, 64)).to(device)}
policy.compute(dummy_input)


from skrl.resources.preprocessors.torch import RunningStandardScaler
from gymnasium import spaces
##########################


def main():
    global key_input

    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
    )
        pass

    #print("resume_path: ", resume_path)
    # resume_path:  /home/kimbring2/isaac_lab_jetbot_orin/logs/skrl/jetbot_orin_direct/
    # 2026-03-19_09-53-15_ppo_torch/checkpoints/agent_100000.pt
    
    # Debug purpose - For selecting checkpoint manually
    '''
    root_path = '/home/kimbring2/isaac_lab_jetbot_orin/logs/skrl/jetbot_orin_direct'
    folder_name = '2026-04-18_12-50-18_ppo_torch'
    resume_path = os.path.join(root_path, folder_name)
    resume_path = os.path.join(resume_path, "checkpoints")
    epoch = 25000
    file_name = "agent_{}.pt".format(epoch)
    resume_path = os.path.join(resume_path, file_name)
    
    '''
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    '''
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    '''

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    #print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0

    # simulate environment
    # Define the number of features in your observation
    num_observations = env.observation_space.shape[0] 

    # Initialize the scaler
    # 'device' should match your obs (e.g., 'cuda:0' or 'cpu')
    # 1. Define the shape (3 channels, 64 height, 64 width)
    obs_shape = (3, 64, 64)

    # 2. Declare the scaler
    # We pass the shape and specify the torch device and dtype
    state_preprocessor = RunningStandardScaler(size=obs_shape, device='cuda:0')

    # IMPORTANT: If you are loading a trained agent, 
    # you should load the scaler's state as well:
    state_preprocessor.load_state_dict(checkpoint['state_preprocessor'])

    # Set to eval mode for inference so it doesn't update mean/var statistics
    state_preprocessor.eval()

    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            #print("runner.agent._state_preprocessor: ", runner.agent._state_preprocessor)

            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            
            # 1. PREPROCESSING: Scale the observations before passing to the policy
            # We use the agent's internal preprocessor if it exists
            if hasattr(runner.agent, "state_preprocessor") and runner.agent.state_preprocessor is not None:
                # Note: state_preprocessor expects a tensor and returns a normalized one
                processed_obs = runner.agent.state_preprocessor(obs)
            else:
                processed_obs = obs

            # Using RunningStandardScaler of skrl for part below
            #processed_obs = runner.agent._state_preprocessor(obs)
            #print("obs.shape: ", obs.shape)
            processed_obs = state_preprocessor(obs)

            with torch.no_grad():
                action, _, _ = policy.act({"states": processed_obs.to('cpu')}, role="policy")

            # Convert the tensor to a numpy array for the robot motors
            action_np = action.cpu().numpy()[0]
            #print("action_np: ", action_np)
            #action_np = np.clip(action_np, -7.5, 7.5) / 30.0
            actions = torch.tensor([float(action_np[0]), float(action_np[1])], device='cuda:0')

            # env stepping
            # print information from the sensors
            # Debug purpose - For rendering camera sensor image
            left_camera_image = env.scene["left_camera"].data.output["rgb"]
            #print(f"Image saved! Mean pixel value: {left_camera_image.cpu().numpy()[0].mean()}")
            #cv2.imshow('Left Camera', left_camera_image.cpu().numpy()[0] / 255.0)

            right_camera_image = env.scene["right_camera"].data.output["rgb"]
            #cv2.imshow('Right Camera', right_camera_image.cpu().numpy()[0] / 255.0)

            weights = torch.tensor([0.2989, 0.5870, 0.1140], device='cuda:0').view(1, 3, 1, 1)

            # 2. Pre-process Left Camera
            # Shape changes: [B, H, W, 3] -> [B, 3, H, W]
            left_camera_image = left_camera_image.permute(0, 3, 1, 2).float()
            left_camera_image = F.interpolate(left_camera_image, size=(64, 64), mode='bilinear', align_corners=False)
            left_camera_image = left_camera_image / 255.0
            left_gray = (left_camera_image * weights).sum(dim=1, keepdim=True)

            left_gray_resized = left_gray.cpu().numpy()[0]
            left_gray_resized = np.transpose(left_gray_resized, axes=(1, 2, 0))
            left_gray_resized = np.squeeze(left_gray_resized) * 255.0
            left_gray_resized = left_gray_resized.astype(np.uint8) 

            # 3. Pre-process Right Camera
            right_camera_image = right_camera_image.permute(0, 3, 1, 2).float()
            right_camera_image = F.interpolate(right_camera_image, size=(64, 64), mode='bilinear', align_corners=False)
            right_camera_image = right_camera_image / 255.0
            right_gray = (right_camera_image * weights).sum(dim=1, keepdim=True)

            right_gray_resized = right_gray.cpu().numpy()[0]
            right_gray_resized = np.transpose(right_gray_resized, axes=(1, 2, 0))
            right_gray_resized = np.squeeze(right_gray_resized) * 255.0
            right_gray_resized = right_gray_resized.astype(np.uint8) 

            #cv2.imshow('Left Camera', left_gray_resized)
            #cv2.imshow('Right Camera', right_gray_resized)
            #cv2.waitKey(1) # Required for the window to refresh
            
            # Debug purpose - For keyboard manual control
            '''
            action_value = 7.5
            if key_input == 'w':
                actions = torch.tensor([action_value, action_value], device='cuda:0')
            elif key_input == 'a':
                actions = torch.tensor([-action_value, action_value], device='cuda:0')
            elif key_input ==  'd':
                actions = torch.tensor([action_value, -action_value], device='cuda:0')
            elif key_input == 's':
                actions = torch.tensor([-action_value, -action_value], device='cuda:0')
            else:
                actions = torch.tensor([0.0, 0.0], device='cuda:0')
            '''

            #actions = torch.tensor([4.0, 0.0], device='cuda:0')
            #actions = torch.clamp(actions, min=-7.5, max=7.5)
            #print("actions: ", actions)

            obs, _, _, _, _ = env.step(actions)

            img_tensor = obs[0].view(3, 64, 64)
            img_tensor_left = img_tensor[0].view(1, 64, 64)
            img_tensor_right = img_tensor[2].view(1, 64, 64)

            # 2. Transpose to (Height, Width, Channels) for OpenCV/Matplotlib
            img_left_np = img_tensor_left.cpu().numpy().transpose(1, 2, 0)
            img_right_np = img_tensor_right.cpu().numpy().transpose(1, 2, 0)

            img_left_resized = cv2.resize(img_left_np, (256, 256), interpolation=cv2.INTER_LINEAR)
            img_right_resized = cv2.resize(img_right_np, (256, 256), interpolation=cv2.INTER_LINEAR)

            # 3. Check for noise visually
            #cv2.imshow('Left Camera', img_left_resized)
            #cv2.imshow('Right Camera', img_right_resized)
            #cv2.waitKey(1)
            
        if args_cli.video:
            timestep += 1
            
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    
    # close sim app
    simulation_app.close()
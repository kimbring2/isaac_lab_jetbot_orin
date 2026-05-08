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
print("checkpoint: ", checkpoint)

policy.load_state_dict(checkpoint['policy']) # skrl stores it under 'policy'
policy.eval()

dummy_input = {"states": torch.zeros((1, 3, 64, 64)).to(device)}
policy.compute(dummy_input)
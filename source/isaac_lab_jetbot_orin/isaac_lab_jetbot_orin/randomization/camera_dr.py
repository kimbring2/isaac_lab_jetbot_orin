import os
import yaml
import torch
import numpy as np
import logging
import glob
import re
import random
import omni.usd


def randomize_camera_parameters(
    env, 
    env_ids: torch.Tensor, 
    focal_length_range: tuple = (0.8, 1.8),
    focus_dist_range: tuple = (5.0, 10.0)  # Added range for Focus Distance
) -> None:
    """Randomizes the focal length and focus distance of cameras."""
    
    camera_0_path_template = "/World/envs/env_{}/Robot/Body/tiled_camera_0"
    camera_1_path_template = "/World/envs/env_{}/Robot/Body/tiled_camera_1" # Fixed index from 0 to 1

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    stage = omni.usd.get_context().get_stage()

    for env_idx in env_ids:
        camera_0_path = camera_0_path_template.format(env_idx)
        camera_1_path = camera_1_path_template.format(env_idx)

        camera_0_prim = stage.GetPrimAtPath(camera_0_path)
        camera_1_prim = stage.GetPrimAtPath(camera_1_path) # Fixed path reference

        # Check validity
        if not camera_0_prim.IsValid() or not camera_1_prim.IsValid():
            continue

        # Generate random values
        focal_length = random.uniform(focal_length_range[0], focal_length_range[1])
        focus_distance = random.uniform(focus_dist_range[0], focus_dist_range[1])

        # Get Attributes
        # Note: USD attributes for cameras typically use camelCase: 'focalLength', 'focusDistance'
        attrs_to_set = {
            "focalLength": focal_length,
            "focusDistance": focus_distance
        }

        for prim in [camera_0_prim, camera_1_prim]:
            for attr_name, value in attrs_to_set.items():
                attr = prim.GetAttribute(attr_name)
                if attr.IsValid():
                    attr.Set(value)
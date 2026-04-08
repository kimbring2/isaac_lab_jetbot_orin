import torch
import numpy as np
import isaacsim.core.utils.prims as prim_utils


def randomize_track_scale(
    env, 
    env_ids,
    scale_range: tuple = (0.05, 0.1),
) -> None:
    # Generate random scales
    #print("randomize_track_scale()")
    prim_path = "/World/envs/env_0/Track" 

    # Get the attribute
    #current_scale = prim_utils.get_prim_attribute_value(prim_path, "xformOp:scale")
    #print(f"Current Track Scale: {current_scale}")

    # 1. Generate a random scale (e.g., between 0.04 and 0.08)
    random_scale_val = np.random.uniform(0.1, 0.2)
    new_scale = (random_scale_val, random_scale_val, random_scale_val)
    prim_utils.set_prim_attribute_value(prim_path, "xformOp:scale", new_scale)
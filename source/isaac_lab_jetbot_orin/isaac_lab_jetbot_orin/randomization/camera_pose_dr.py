import torch
import random
import omni.usd
from pxr import Gf


def randomize_camera_pose(
    env, 
    env_ids: torch.Tensor, 
    pos_jitter: float = 0.05,      # +/- 5mm
    rot_jitter_deg: float = 5.0     # +/- 2 degrees
) -> None:
    """Randomizes focal length, focus distance, position, and rotation."""
    
    camera_0_path_template = "/World/envs/env_{}/Robot/Body/tiled_camera_0"
    camera_1_path_template = "/World/envs/env_{}/Robot/Body/tiled_camera_1"

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    stage = omni.usd.get_context().get_stage()

    for env_idx in env_ids:
        cam_paths = [camera_0_path_template.format(env_idx), camera_1_path_template.format(env_idx)]
        
        for path in cam_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue

            # --- 1. Position Jitter ---
            # Get current base position to add noise
            pos_attr = prim.GetAttribute("xformOp:translate")
            if pos_attr.IsValid():
                current_pos = pos_attr.Get()
                noise_pos = Gf.Vec3d(
                    random.uniform(-pos_jitter, pos_jitter),
                    random.uniform(-pos_jitter, pos_jitter),
                    random.uniform(-pos_jitter, pos_jitter)
                )
                pos_attr.Set(current_pos + noise_pos)

            # --- 2. Rotation Jitter (Euler) ---
            # Most Isaac Lab assets use xformOp:rotateXYZ or xformOp:orient (Quaternion)
            # We'll check for rotateXYZ as it's easiest for small jitters
            rot_attr = prim.GetAttribute("xformOp:rotateXYZ")
            if rot_attr.IsValid():
                current_rot = rot_attr.Get() # Gf.Vec3f
                noise_rot = Gf.Vec3f(
                    random.uniform(-rot_jitter_deg, rot_jitter_deg),
                    random.uniform(-rot_jitter_deg, rot_jitter_deg),
                    random.uniform(-rot_jitter_deg, rot_jitter_deg)
                )
                rot_attr.Set(current_rot + noise_rot)
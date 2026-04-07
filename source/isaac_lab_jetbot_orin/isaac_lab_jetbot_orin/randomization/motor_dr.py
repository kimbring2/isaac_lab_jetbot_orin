import random
import torch
import omni.usd
from pxr import PhysxSchema


def randomize_motor_parameters(
    env, 
    env_ids: torch.Tensor, 
    damping_range: tuple = (100.0, 200.0),
    stiffness_range: tuple = (0.0, 10.0)
) -> None:
    """Randomizes the damping and stiffness of the wheel drive joints."""
    
    # Path templates for your joints
    joint_paths = [
        "/World/envs/env_{}/Robot/left_wheel_joint",
        "/World/envs/env_{}/Robot/right_wheel_joint"
    ]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    stage = omni.usd.get_context().get_stage()

    for env_idx in env_ids:
        for path_template in joint_paths:
            # Generate new random values for this specific environment
            new_damping = random.uniform(damping_range[0], damping_range[1])
            new_stiffness = random.uniform(stiffness_range[0], stiffness_range[1])

            joint_path = path_template.format(env_idx)
            joint_prim = stage.GetPrimAtPath(joint_path)

            if not joint_prim.IsValid():
                continue

            # Drive attributes in USD follow a specific naming convention:
            # drive:[name]:[parameter]
            # Usually, wheel drives are named "angular" or "drive"
            damping_attr = joint_prim.GetAttribute("drive:angular:damping")
            stiffness_attr = joint_prim.GetAttribute("drive:angular:stiffness")

            if damping_attr.IsValid():
                damping_attr.Set(new_damping)
            
            #if stiffness_attr.IsValid():
            #    stiffness_attr.Set(new_stiffness)
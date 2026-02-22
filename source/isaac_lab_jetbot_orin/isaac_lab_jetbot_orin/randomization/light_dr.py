import torch
from isaacsim.core.utils.prims import get_prim_at_path
from pxr import UsdLux, Gf

def randomize_lights(env, env_ids, intensity_range=(500.0, 5000.0),
                     color_range=((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))):
    """Randomize per‑env lights for Isaac Lab 2.3.0 / Isaac Sim 5.1.
    Args:
        env: DirectRLEnv instance (self in your task).
        env_ids: 1D tensor of env indices to randomize.
        intensity_range: (min, max) intensity.
        color_range: ((r_min, g_min, b_min), (r_max, g_max, b_max)).
    """
    device = env.device
    env_ids = env_ids.to(device)
    num = env_ids.shape[0]

    # Sample intensities and colors in torch
    min_I, max_I = intensity_range
    intensities_1 = torch.rand(num, device=device) * (max_I - min_I) + min_I
    intensities_2 = torch.rand(num, device=device) * (max_I - min_I) + min_I
    intensities_3 = torch.rand(num, device=device) * (max_I - min_I) + min_I
    intensities_4 = torch.rand(num, device=device) * (max_I - min_I) + min_I
    #print("intensities: {0}, {1}, {2}, {3}".format(intensities_1, intensities_2, intensities_3, intensities_4))

    min_c = torch.tensor(color_range[0], device=device)
    max_c = torch.tensor(color_range[1], device=device)
    colors_1 = torch.rand(num, 3, device=device) * (max_c - min_c) + min_c
    colors_2 = torch.rand(num, 3, device=device) * (max_c - min_c) + min_c
    colors_3 = torch.rand(num, 3, device=device) * (max_c - min_c) + min_c
    colors_4 = torch.rand(num, 3, device=device) * (max_c - min_c) + min_c
    #print("colors: {0}, {1}, {2}, {3}".format(colors_1, colors_2, colors_3, colors_4))
    #print("\n")

    # Apply to each env light
    for i, env_id in enumerate(env_ids.tolist()):
        prim_path_1 = f"/World/envs/env_{env_id}/Light1"
        prim_path_2 = f"/World/envs/env_{env_id}/Light2"
        prim_path_3 = f"/World/envs/env_{env_id}/Light3"
        prim_path_4 = f"/World/envs/env_{env_id}/Light4"

        prim_1 = get_prim_at_path(prim_path_1)
        prim_2 = get_prim_at_path(prim_path_2)
        prim_3 = get_prim_at_path(prim_path_3)
        prim_4 = get_prim_at_path(prim_path_4)
        if not prim_1 or not prim_2 or not prim_3 or not prim_4:
            continue

        light_1 = UsdLux.BoundableLightBase(prim_1)
        light_2 = UsdLux.BoundableLightBase(prim_2)
        light_3 = UsdLux.BoundableLightBase(prim_3)
        light_4 = UsdLux.BoundableLightBase(prim_4)

        light_1.CreateIntensityAttr().Set(float(intensities_1[i].item()))
        light_2.CreateIntensityAttr().Set(float(intensities_2[i].item()))
        light_3.CreateIntensityAttr().Set(float(intensities_3[i].item()))
        light_4.CreateIntensityAttr().Set(float(intensities_4[i].item()))

        col_1 = colors_1[i]
        col_2 = colors_2[i]
        col_3 = colors_3[i]
        col_4 = colors_4[i]

        light_1.CreateColorAttr().Set(
            Gf.Vec3f(float(col_1[0].item()), float(col_1[1].item()), float(col_1[2].item()))
        )

        light_2.CreateColorAttr().Set(
            Gf.Vec3f(float(col_2[0].item()), float(col_2[1].item()), float(col_2[2].item()))
        )

        light_3.CreateColorAttr().Set(
            Gf.Vec3f(float(col_3[0].item()), float(col_3[1].item()), float(col_3[2].item()))
        )

        light_4.CreateColorAttr().Set(
            Gf.Vec3f(float(col_4[0].item()), float(col_4[1].item()), float(col_4[2].item()))
        )
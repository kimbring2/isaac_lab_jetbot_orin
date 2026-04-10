import os
import yaml
import torch
import numpy as np
import logging
import glob
import re
import random
from pxr import UsdShade, Sdf, Gf
from isaaclab.managers import SceneEntityCfg
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid


def change_curtain_texture(
    env,
    env_ids
):
    for i, env_id in enumerate(env_ids.tolist()):
        # 1. Base path
        track_base_path = f"/World/envs/env_{env_id}/Track/simple_curved_track/Visuals"
        mat_path = f"{track_base_path}/Looks/curtain"
        
        # 2. Get the material prim
        stage = env.sim.stage
        mat_prim = stage.GetPrimAtPath(mat_path)
        
        if not mat_prim.IsValid():
            return

        # 3. Create a random RGB color
        random_rgb = Gf.Vec3f(
            np.random.random(), 
            np.random.random(), 
            np.random.random()
        )

        # 4. Apply to the diffuse color attribute
        # Note: OmniPBR usually uses 'inputs:diffuse_color_constant'
        shader_prim = stage.GetPrimAtPath(f"{mat_path}/Shader")
        if shader_prim.IsValid():
            shader_prim.GetAttribute("inputs:diffuse_color_constant").Set(random_rgb)


def change_track_texture(
    env,
    env_ids,
    new_straight_texture_path: str = "/home/kimbring2/.../textures/straight_2.png" # Full path
):
    stage = env.sim.stage
  
    # 1. Define the directory path
    new_straight_texture_paths_dir = os.path.join(os.getcwd(), new_straight_texture_path)

    # 2. Gather all PNG files in that folder
    # glob.glob returns a list of full strings: ['/path/to/straight_1.png', '/path/to/straight_2.png', ...]
    available_straight_textures = glob.glob(os.path.join(new_straight_texture_paths_dir, "*.png"))

    for i, env_id in enumerate(env_ids.tolist()):
        # 3. Select one path randomly
        if available_straight_textures:
            new_straight_texture_path = np.random.choice(available_straight_textures)
        else:
            new_straight_texture_path = None
            print(f"Warning: No textures found in {new_straight_texture_paths_dir}")

        straight_filename = new_straight_texture_path.split("/")[-1]
        straight_file_number = re.search(r'\d+', straight_filename).group()

        new_curve_texture_path = "/".join(new_straight_texture_path.split("/")[:-2]) + "/curve_left" + "/curve_left_{}.png".format(straight_file_number)
        new_black_texture_path = "/".join(new_straight_texture_path.split("/")[:-2]) + "/black" + "/black_tile_{}.png".format(straight_file_number)

        # 1. Base path for the Track in a specific environment
        track_base_path = f"/World/envs/env_{env_id}/Track/simple_curved_track/Visuals"
        
        # 2. Get the specific material prim. 
        # Adjust this path if 'straight_8' is bound to a different material than 'Looks/straight_1'
        straight_mat_path = f"{track_base_path}/Looks/straight_1"
        curve_mat_path = f"{track_base_path}/Looks/curve_left_1"
        black_mat_path = f"{track_base_path}/Looks/black_tile"
        
        # Verify the material exists
        if not is_prim_path_valid(straight_mat_path):
            # carb.log_warn(f"Material not found at: {mat_path}")
            return

        if not is_prim_path_valid(curve_mat_path):
            # carb.log_warn(f"Material not found at: {mat_path}")
            return

        if not is_prim_path_valid(black_mat_path):
            # carb.log_warn(f"Material not found at: {mat_path}")
            return

        straight_mat_prim = get_prim_at_path(straight_mat_path)
        curve_mat_prim = get_prim_at_path(curve_mat_path)
        black_mat_prim = get_prim_at_path(black_mat_path)

        # We can skip the 'Looks/straight_1/Shader' step and set the input directly
        # on the material if it has been correctly connected, but setting it on
        # the Shader prim is the most direct approach based on image_1.png
        straight_shader_path = f"{straight_mat_path}/Shader"
        if not is_prim_path_valid(straight_shader_path):
            return

        curve_shader_path = f"{curve_mat_path}/Shader"
        if not is_prim_path_valid(curve_shader_path):
            return

        black_shader_path = f"{black_mat_path}/Shader"
        if not is_prim_path_valid(black_shader_path):
            return
            
        straight_shader_prim = get_prim_at_path(straight_shader_path)
        straight_shader = UsdShade.Shader(straight_shader_prim)

        curve_shader_prim = get_prim_at_path(curve_shader_path)
        curve_shader = UsdShade.Shader(curve_shader_prim)

        black_shader_prim = get_prim_at_path(black_shader_path)
        black_shader = UsdShade.Shader(black_shader_prim)
        
        # 3. Get the existing input (as shown in your screenshot)
        # The property is 'inputs:diffuse_texture'
        straight_tex_input = straight_shader.GetInput("diffuse_texture")
        curve_tex_input = curve_shader.GetInput("diffuse_texture")
        black_tex_input = black_shader.GetInput("diffuse_texture")
        
        # 4. If it exists, set the new texture path (MUST use Sdf.AssetPath)
        if straight_tex_input:
            if os.path.exists(new_straight_texture_path):
                # Set the new path
                straight_tex_input.Set(Sdf.AssetPath(new_straight_texture_path))
                #print(f"Texture changed for: {straight_shader_path}")
            else:
                print(f"Error: Texture file not found at {new_straight_texture_path}")

        if curve_tex_input:
            if os.path.exists(new_curve_texture_path):
                # Set the new path
                curve_tex_input.Set(Sdf.AssetPath(new_curve_texture_path))
            else:
                print(f"Error: Texture file not found at {new_curve_texture_path}")

        if black_tex_input:
            if os.path.exists(new_black_texture_path):
                # Set the new path
                black_tex_input.Set(Sdf.AssetPath(new_black_texture_path))
            else:
                print(f"Error: Texture file not found at {new_black_texture_path}")
# Isaac Lab Duckietown Projects

## Overview
<img src="images/isaaclab_duckietown_demo.gif" width="720">

A project to train a vehicle-type robot to drive well in a duckietown environment using deep reinforcement learning.

## Installation

- Install Isaac Lab 4.5 version by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
    ```bash
        git clone https://github.com/kimbring2/isaac_lab_jetbot_orin.git
    ```

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    python -m pip install -e source/isaac_lab_jetbot_orin
    ```

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        ```bash
        python scripts/list_envs.py
        ```

    - Running a training task:
        ```bash
        python scripts/skrl/train.py --task=Isaac-Lab-Jetbot-Orin-Direct-v0 --enable_cameras --num_envs 1
        ```

    - Running a playing task:
        ```bash
        python scripts/skrl/play.py --task=Isaac-Lab-Jetbot-Orin-Direct-v0 --num_envs 1 --enable_cameras
        ```
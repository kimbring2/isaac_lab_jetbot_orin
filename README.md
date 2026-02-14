# Isaac Lab Duckietown Projects

## Overview
<img src="images/isaaclab_duckietown_demo.gif" width="720">

A project to train a vehicle-type robot to drive well in a duckietown environment using deep reinforcement learning.

**Key Features:**
- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, jetbot, duckietown, isaaclab

## Installation

- Install Isaac Lab 4.5 version by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
    ```bash
        git clone https://github.com/kimbring2/isaac_lab_jetbot_orin.git
    ```

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/isaac_lab_jetbot_orin

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:
        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/skrl/train.py --task=Isaac-Lab-Jetbot-Orin-Direct-v0 --enable_cameras --num_envs 4
        ```
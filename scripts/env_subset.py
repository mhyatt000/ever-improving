from __future__ import annotations

import json
from collections import defaultdict
from pprint import pprint
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
import simpler_env
from gymnasium import logger, spaces
from gymnasium.core import (ActionWrapper, Env, ObservationWrapper,
                            RewardWrapper, Wrapper)
from gymnasium.spaces.box import Box
from gymnasium.spaces.dict import Dict
from gymnasium.spaces.space import Space
from omegaconf import OmegaConf
from omegaconf import OmegaConf as OC
from scipy.ndimage import zoom
from simpler_env.utils.env.observation_utils import \
    get_image_from_maniskill2_obs_dict
from tqdm import tqdm

import improve
import improve.config.resolver
import improve.wrapper.dict_util as du
import improve.wrapper.residualrl as rrl


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):
    pprint(OmegaConf.to_container(cfg, resolve=True))

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    allinstructions = set()

    EPISODES = 1000
    MAX_STEPS = 2000

    # for t in hist:
    env = rrl.make(cfg.env)

    episode_infos = []

    # initially add empty list to create file and initialize
    with open("start_infos.json", "w") as file:
        json.dump(episode_infos, file, indent=4)

    for _ in tqdm(range(EPISODES),leave=False):
        obs, initial_info = env.reset()

        # perform transformations to allow for json file saving
        initial_info["episode_source_obj_init_pose_wrt_robot_base"] = list(
            initial_info["episode_source_obj_init_pose_wrt_robot_base"].__getstate__()
        )
        initial_info["episode_target_obj_init_pose_wrt_robot_base"] = list(
            initial_info["episode_target_obj_init_pose_wrt_robot_base"].__getstate__()
        )

        done=False
        while not done:
            # zero action
            action = np.zeros(env.action_space.shape)
            # random action
            # env.action_space.sample()

            observation, reward, success, truncated, info = env.step(action)

            allinstructions.add(env.instruction)
            if not (env.instruction in allinstructions):
                print(env.instruction)

            if truncated or success:
                initial_info["success"] = success
                done=True

        with open("start_infos.json", "r") as file:
            episode_infos = json.load(file)

        episode_infos.append(initial_info)

        with open("start_infos.json", "w") as file:
            json.dump(episode_infos, file, indent=4)

    env.close()

    print(allinstructions)


if __name__ == "__main__":
    main()

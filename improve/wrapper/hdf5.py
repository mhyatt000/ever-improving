from __future__ import annotations

import collections
import os
import os.path as osp
import time
from datetime import datetime

import gym
import gymnasium as gym
import h5py
import hydra
import improve
import improve.hydra.resolver
import improve.wrapper.dict_util as du
import numpy as np
from gymnasium.core import Wrapper
from omegaconf import OmegaConf
from omegaconf import OmegaConf as OC
from tqdm import tqdm

HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "datasets", "simpler")

"""
    self.store(obs, reward, terminated, truncated, action, info)
  File "/home/zero-shot/mhyatt000/ever-improving/improve/wrapper/hdf5.py", line 98, in store
    info_group.create_dataset(key, data=np.array(value))
  File "/home/zero-shot/miniconda3/envs/improve/lib/python3.11/site-packages/h5py/_hl/group.py", line 183, in create_dataset
    dsid = dataset.make_new_dset(group, shape, dtype, data, name, **kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zero-shot/miniconda3/envs/improve/lib/python3.11/site-packages/h5py/_hl/dataset.py", line 86, in make_new_dset
    tid = h5t.py_create(dtype, logical=1)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h5py/h5t.pyx", line 1663, in h5py.h5t.py_create
  File "h5py/h5t.pyx", line 1687, in h5py.h5t.py_create
  File "h5py/h5t.pyx", line 1747, in h5py.h5t.py_create
TypeError: Object dtype dtype('O') has no native HDF5 equivalent
"""


class HDF5LoggerWrapper(Wrapper):

    def __init__(self, env, task, rootdir=DATA_DIR, id=None, cfg=None):
        super(HDF5LoggerWrapper, self).__init__(env)

        now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.rootdir = rootdir
        self.fname = osp.join(self.rootdir, f"dataset_{now}_{task}.h5")
        self.file = None
        self.dataset_info_group = None
        self.episode_group = None
        self.step_group = None
        self.counter = 0

        os.makedirs(self.rootdir, exist_ok=True)  # Ensure the directory exists

        if os.path.exists(self.fname):
            self.file = h5py.File(self.fname, "a", libver="latest")
        else:
            self.file = h5py.File(self.fname, "w", libver="latest")

    def reset(self, **kwargs):

        self.wait_for_break()

        self.dataset_info_group = self.file.require_group("dataset_info")

        # Create a new group with the current date and time
        now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.episode_group = self.file.create_group(f"ep_{now}")

        self.step_group = self.episode_group.create_group("steps")
        obs, info = self.env.reset(**kwargs)

        self.obs = obs
        self.counter = 0
        return obs, info

    def create_datasets_from_dict(self, group, data_dict):
        for key, value in data_dict.items():
            if isinstance(value, (dict, collections.OrderedDict)):
                value = du.todict(value)
                subgroup = group.create_group(key)
                self.create_datasets_from_dict(subgroup, value)
            else:
                group.create_dataset(key, data=np.array(value))

    def store(self, obs, reward, terminated, truncated, action, info):

        assert self.task is not None, "task must be set before storing data"

        step = {
            "observation": self.obs,
            # "next_observation": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "action": action,
            "info": du.todict(info),
        }
        # we want to store the observation that conditioned the action
        self.obs = obs

        step_dataset = self.step_group.create_group(f"step_{self.counter}")

        self.create_datasets_from_dict(step_dataset, step)
        self.counter += 1

        # add dataset information for that episode
        if terminated or truncated:
            episode_info = {
                "task": self.task,
                "n_steps": self.counter,
                "success": reward > 0.0,
            }

            episode_info_group = self.dataset_info_group.create_group(
                self.episode_group.name.split("/")[-1]
            )
            for key, value in episode_info.items():
                episode_info_group.create_dataset(key, data=value)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.store(obs, reward, terminated, truncated, action, info)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.file.close()
        self.env.close()

    def wait_for_break(self):
        desc = "press exit to break the loop"
        try:
            for _ in tqdm(range(30), desc=desc, leave=False):
                time.sleep(0.1)
            return False
        except KeyboardInterrupt:
            print("break")
            self.close()
            return True


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):

    from improve.env import make_env

    env = make_env(cfg)()
    env = HDF5LoggerWrapper(env, task=cfg.env.foundation.task)

    """
    obs, info = env.reset()
    print(info)
    env.close()
    quit()
    """

    desc = "collecting data... do not disturb"
    for i in tqdm(range(int(1e2)), desc=desc, leave=False):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            zero = np.zeros_like(action)
            obs, reward, terminated, truncated, info = env.step(zero)

            done = terminated or truncated
        if env.wait_for_break():
            break

    env.close()


if __name__ == "__main__":
    main()

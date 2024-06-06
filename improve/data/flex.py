import os
from multiprocessing import Lock
import random
import os.path as osp
from pprint import pprint

import h5py
import hydra
import functools
import improve
from improve.wrapper.dict_util import apply_both
import improve.config.resolver
import improve.wrapper.dict_util as du
import numpy as np
import torch
from omegaconf import OmegaConf as OC
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader as Dataloader

HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "datasets", "simpler")


class HDF5IterDataset(IterableDataset):

    def __init__(self, root_dir=DATA_DIR, loop=False, n_steps=1):
        super(HDF5IterDataset, self).__init__()

        self.root_dir = root_dir
        
        self.fnames = [osp.join(root_dir, 'dataset.h5')]
        if not loop:
            self.fnames = iter(self.fnames)
            
        self.n_steps = n_steps
        self.lock = Lock()
        torch.multiprocessing.set_sharing_strategy('file_system')
        
        # self.fnames = [
        #     osp.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".h5")
        # ][:-1]
        # if not loop:
        #     self.fnames = iter(self.fnames)

        # self.n_steps = n_steps
        # self.lock = Lock()
        # torch.multiprocessing.set_sharing_strategy('file_system')

    @staticmethod
    def to_tensor(h):
        if isinstance(h, h5py.Dataset):
            data = h[()]
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data)
            else:
                return torch.tensor(data)

        elif isinstance(h, (int, float, bool, str)):
            return torch.tensor(h)
        else:
            raise TypeError(f"Unsupported type: {type(h)}")

    @staticmethod
    def extract(h):
        if isinstance(h, h5py.Group):
            return {k: HDF5IterDataset.extract(v) for k, v in h.items()}
        else:
            return HDF5IterDataset.to_tensor(h)

    def _iter_by_step(self):
        for fname in self.fnames:
            with h5py.File(fname, "r", libver='latest', swmr=True) as f:
                # go through each episode (skip the first info section)
                for episode in [x for x in f.keys() if x != 'dataset_info']:
                    steps = list(f[episode]["steps"].keys())
                    steps.sort(key = lambda x: int(x.split('_')[1]))
                    for key in steps:
                        step = f[episode]["steps"][key]
                        yield HDF5IterDataset.extract(step)

    def _iter_by_trajectory(self):
        for fname in self.fnames:
            with h5py.File(fname, "r", libver='latest', swmr=True) as f:
                trajectory = []
                for episode in [x for x in f.keys() if x != 'dataset_info']:
                    
                    steps = list(f[episode]["steps"].keys())
                    steps.sort(key = lambda x: int(x.split('_')[1]))
                    steps = steps[random.randint(0, self.n_steps - 1) :]
                    for key in steps:
                        step = f[episode]["steps"][key]
                        trajectory.append(HDF5IterDataset.extract(step))
                        if len(trajectory) == self.n_steps:
                            trajectory = [du.apply(x, lambda x: torch.unsqueeze(x,0)) for x in trajectory]
                            yield functools.reduce(lambda a, b: apply_both(a, b, torch.cat), trajectory)
                            trajectory = []
                    trajectory = [] # reset trajectory if not enough steps

    def __iter__(self):
        if self.n_steps == 1:
            yield from self._iter_by_step()
        else:
            yield from self._iter_by_trajectory()


def inspect(data):
    x = du.apply(data, lambda x: (x.dtype, x.shape))
    pprint(x)
    quit()


def main():

    n_success = 0
    n = 0

    # D = HDF5IterDataset(DATA_DIR, loop=False, n_steps=10)
    D = HDF5IterDataset(DATA_DIR, loop=False, n_steps=10)
    dataset = Dataloader(D, batch_size=1, num_workers=4)
    
    for data in dataset:
        print(data)

    # print(D.fnames)
    # for data in D:
        # inspect(data)

        # print(data['reward'].item())
        # print(data['info']['is_success'].item())
        # quit()

        # data['info']['is_success'] and data['terminated']
        # n_success += data["reward"].item()
        # n += 1
        # print(f"n_success: {n_success} | n: {n}")


if __name__ == "__main__":
    main()

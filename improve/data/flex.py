import functools
import os
import os.path as osp
import random
from multiprocessing import Lock
from pprint import pprint

import h5py
import hydra
import improve
import improve.config.resolver
import improve.wrapper.dict_util as du
import numpy as np
import torch
from improve.wrapper.dict_util import apply_both
from omegaconf import OmegaConf as OC
from torch.utils.data import DataLoader as Dataloader
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset

HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "datasets", "simpler")


class HDF5Dataset(Dataset):

    def __init__(self, root_dir=DATA_DIR, n_steps=1):
        super(HDF5Dataset, self).__init__()

        self.root_dir = root_dir

        self.fnames = [osp.join(root_dir, "dataset.h5")]
        self.n_steps = n_steps

        # shuffle the dataset using seq length
        seq_len = n_steps

        idxs = []
        for fname in self.fnames:
            with h5py.File(fname, "r", libver="latest", swmr=True) as f:
                # go through each episode (skip the first info section)
                for episode in f["dataset_info"].keys():
                    episode_len = f["dataset_info"][episode]["n_steps"][()]
                    n, rem = divmod(episode_len, seq_len)

                    # add last seq with reward to idxs
                    if n > 0 and rem > 0:
                        idxs.append((episode, (-seq_len, episode_len)))

                    # add remaining windows to idxs
                    offset = random.randint(0, rem)

                    for i in range(n):
                        idxs.append((episode, (offset + i, offset + i + seq_len)))

        self.idxs = idxs

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
            return {k: HDF5Dataset.extract(v) for k, v in h.items()}
        else:
            return HDF5Dataset.to_tensor(h)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        for fname in self.fnames:
            with h5py.File(fname, "r", libver="latest", swmr=True) as f:
                episode, (start, end) = self.idxs[idx]
                steps = list(f[episode]["steps"].keys())
                steps.sort(key=lambda x: int(x.split("_")[1]))

                trajectory = []
                for key in steps[start:end]:
                    step = f[episode]["steps"][key]
                    trajectory.append(HDF5Dataset.extract(step))

                trajectory = [
                    du.apply(x, lambda x: torch.unsqueeze(x, 0)) for x in trajectory
                ]
                return functools.reduce(
                    lambda a, b: apply_both(a, b, torch.cat), trajectory
                )


class HDF5IterDataset(IterableDataset):

    def __init__(self, root_dir=DATA_DIR, loop=False, n_steps=1):
        super(HDF5IterDataset, self).__init__()

        self.root_dir = root_dir

        self.fnames = [osp.join(root_dir, "dataset.h5")]
        if not loop:
            self.fnames = iter(self.fnames)

        self.n_steps = n_steps

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
            with h5py.File(fname, "r", libver="latest", swmr=True) as f:
                # go through each episode (skip the first info section)
                for episode in [x for x in f.keys() if x != "dataset_info"]:
                    steps = list(f[episode]["steps"].keys())
                    steps.sort(key=lambda x: int(x.split("_")[1]))
                    for key in steps:
                        step = f[episode]["steps"][key]
                        yield HDF5IterDataset.extract(step)

    def _iter_by_trajectory(self):
        for fname in self.fnames:
            with h5py.File(fname, "r", libver="latest", swmr=True) as f:
                trajectory = []
                for episode in [x for x in f.keys() if x != "dataset_info"]:

                    steps = list(f[episode]["steps"].keys())
                    steps.sort(key=lambda x: int(x.split("_")[1]))
                    steps = steps[random.randint(0, self.n_steps - 1) :]
                    for key in steps:
                        step = f[episode]["steps"][key]
                        trajectory.append(HDF5IterDataset.extract(step))
                        if len(trajectory) == self.n_steps:
                            trajectory = [
                                du.apply(x, lambda x: torch.unsqueeze(x, 0))
                                for x in trajectory
                            ]
                            yield functools.reduce(
                                lambda a, b: apply_both(a, b, torch.cat), trajectory
                            )
                            trajectory = []
                    trajectory = []  # reset trajectory if not enough steps

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

    # n_success = 0
    # n = 0

    # D = HDF5IterDataset(DATA_DIR, loop=False, n_steps=10)
    D = HDF5Dataset(DATA_DIR, n_steps=10)
    loader = Dataloader(D, batch_size=8, num_workers=4)

    batch = next(iter(loader))

    print(len(D))
    quit()

    print(du.apply(batch, lambda x: x.shape))
    quit()

    for data in loader:
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

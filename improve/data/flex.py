import os
from pprint import pprint
import os.path as osp

import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf as OC
from torch.utils.data import IterableDataset

import improve
import improve.config.resolver
import improve.wrapper.dict_util as du

HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "datasets", "simpler")


class HDF5IterDataset(IterableDataset):

    def __init__(self, root_dir, loop=False):
        super(HDF5IterDataset, self).__init__()

        self.root_dir = root_dir
        self.fnames = [
            osp.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".h5")
            ][:-1]
        if not loop:
            self.fnames = iter(self.fnames)

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

    def __iter__(self):
        for i, fname in enumerate(self.fnames):
            with h5py.File(fname, "r") as f:
                for step_key in f["steps"].keys():
                    step = f["steps"][step_key]
                    yield HDF5IterDataset.extract(step)


def inspect(data):
    x = du.apply(data, lambda x: (x.dtype, x.shape))
    pprint(x)
    quit()

def main():

    n_success = 0
    n = 0

    D = HDF5IterDataset(DATA_DIR, loop=False)
    print(D.fnames)
    for data in D:
        inspect(data)

        # print(data['reward'].item())
        # print(data['info']['is_success'].item())
        # quit()

        # data['info']['is_success'] and data['terminated']
        n_success += data['reward'].item()
        n += 1
        print(f"n_success: {n_success} | n: {n}")


if __name__ == "__main__":
    main()

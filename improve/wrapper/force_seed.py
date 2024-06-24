import itertools
import random

import gymnasium as gym
import numpy as np
import simpler_env as simpler
from gymnasium import spaces
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name


class ForceSeedWrapper(gym.Wrapper):
    """
    Force the seed of the environment to be a specific value

    :param env: the environment to wrap
    :param seed: the seed to force
    :param seeds: a list of seeds to cycle through
    :verbose: print the seed that is being forced
    """

    def __init__(self, env, seed=0, seeds=None, verbose=False):
        super().__init__(env)

        self.seed = seed

        if seeds is not None:
            if isinstance(seeds, int):
                seeds = list(range(seeds))
            random.shuffle(seeds)
            self.seeds = itertools.cycle(seeds)
        else:
            self.seeds = None

        self.verbose = verbose

    def reset(self, **kwargs):

        if self.seeds is not None:
            kwargs["seed"] = next(self.seeds)
        else:
            kwargs["seed"] = self.seed

        if self.verbose:
            print(f"{type(self)} forcing seed to {kwargs['seed']}")
        return self.env.reset(**kwargs)

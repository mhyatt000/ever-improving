import os.path as osp
import time
import warnings
from typing import Optional, Tuple

import numpy as np
from improve.wrapper import dict_util as du
from stable_baselines3.common.vec_env.base_vec_env import (VecEnv, VecEnvObs,
                                                           VecEnvStepReturn,
                                                           VecEnvWrapper)


class WandbVecMonitor(VecEnvWrapper):
    """
    A vectorized monitor wrapper for *vectorized* Gym environments,
    it is used to record the episode reward, length, time and other data.

    Some environments like `openai/procgen <https://github.com/openai/procgen>`_
    or `gym3 <https://github.com/openai/gym3>`_ directly initialize the
    vectorized environments, without giving us a chance to use the ``Monitor``
    wrapper. So this class simply does the job of the ``Monitor`` wrapper on
    a vectorized level.

    :param venv: The vectorized environment
    """

    def __init__(self, venv: VecEnv, logger):

        # Avoid circular import
        from stable_baselines3.common.monitor import Monitor, ResultsWriter

        # This check is not valid for special `VecEnv`
        # like the ones created by Procgen, that does follow completely
        # the `VecEnv` interface
        try:
            is_wrapped_with_monitor = venv.env_is_wrapped(Monitor)[0]
        except AttributeError:
            is_wrapped_with_monitor = False

        if is_wrapped_with_monitor:
            warnings.warn(
                "The environment is already wrapped with a `Monitor` wrapper"
                "but you are wrapping it with a `VecMonitor` wrapper, the `Monitor` statistics will be"
                "overwritten by the `VecMonitor` ones.",
                UserWarning,
            )

        VecEnvWrapper.__init__(self, venv)
        self.episode_count = 0
        self.t_start = time.time()

        env_id = None
        if hasattr(venv, "spec") and venv.spec is not None:
            env_id = venv.spec.id

        self.logger = logger
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step_wait(self) -> VecEnvStepReturn:

        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])

        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()

                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "t": round(time.time() - self.t_start, 6),
                }

                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                new_infos[i] = info

        _infos = du.concat(new_infos) if len(new_infos) > 1 else new_infos[0]

        def tofloat(a):
            if isinstance(a, np.ndarray):
                return a.astype(float).mean()
            if isinstance(a, (list, tuple)):
                a = [float(x) for x in a]
                return sum(a) / len(a)
            return float(a)

        _infos = du.apply(_infos, tofloat)

        for k, v in du.flatten(_infos, delim="/").items():
            k = osp.join("stats", "info", k)
            self.logger.record_mean(k, v)

        return obs, rewards, dones, new_infos

    def close(self) -> None:
        return self.venv.close()

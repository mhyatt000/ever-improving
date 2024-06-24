import gymnasium as gym
import numpy as np
import simpler_env as simpler
from gymnasium import spaces
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name


class NoRotationWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.qpos = np.array(
            [
                -0.065322585,
                0.12452538,
                0.47524214,
                1.0814414,
                -0.19315898,
                1.7895244,
                -0.98058003,
                -5.158836e-08,
                2.2535543e-08,
                -0.00285961,
                0.7851361,
            ]
        )

        self.env = env
        self.env.agent.reset(self.qpos)

    def reset(self):
        things = self.env.reset()
        self.env.agent.reset(self.qpos)
        return things

    def step(self, action):

        # these are the yaw pitch roll values
        action[3:6] = 0

        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info

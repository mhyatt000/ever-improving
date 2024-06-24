import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces.dict import Dict
from improve.wrapper import dict_util as du
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from scipy.ndimage import zoom


class FlattenKeysWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        spaces = du.flatten(du.todict(self.observation_space))
        self.observation_space = Dict(spaces)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        observation = du.flatten(du.todict(observation))
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        observation = self.observation(observation)
        return observation, reward, terminated, truncated, info


class FilterKeysWrapper(gym.Wrapper):

    def __init__(self, env, keys=None):
        assert isinstance(
            env, FlattenKeysWrapper
        ), "must be a FlattenKeysWrapper to filter keys"
        super().__init__(env)

        # filter for the desired obs space
        self.keys = keys
        spaces = {k: v for k, v in self.env.observation_space.items() if k in self.keys}
        self.observation_space = Dict(spaces)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        observation = {k: observation[k] for k in self.keys}
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        observation = self.observation(observation)
        return observation, reward, terminated, truncated, info


class GraspDenseRewardWrapper(gym.Wrapper):

    def __init__(self, env, clip=0.2):
        super().__init__(env)
        self.clip = clip

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        reward = self.compute_reward(
            observation, action, reward, terminated, truncated, info
        )
        return observation, reward, terminated, truncated, info

    def compute_reward(self, observation, action, reward, terminated, truncated, info):

        dist = self.obj_wrt_eef()
        dist_reward = np.clip(1 - np.tanh(10 * np.linalg.norm(dist)), -1, self.clip)

        return 10 * reward + ((1e-2) * dist_reward) + (1 * info["is_grasped"])

        # + int(info["lifted_object"])
        # + ( (1e-3) * sum(action) if self.model is not None else 0)  # RP shouldnt help too much

    def obj_wrt_eef(self):
        """Get the object pose with respect to the end-effector frame"""
        return self.obj_pose.p - self.get_tcp().pose.p


def _scale_image(image, scale):

    # TODO can we get rid of batch dim?
    zoom_factors = (scale, scale, 1)
    scaled_image = zoom(image, zoom_factors)
    return scaled_image


def isimg(o):
    if isinstance(o, np.ndarray) and o.ndim == 3:
        return o.shape[-1] in [1, 3, 4]
    return False


class DownscaleImgWrapper(gym.Wrapper):
    """downscale the image observation by a factor
    SHOULD work for any type and combination of images
    """

    def __init__(self, env, downscale):
        super().__init__(env)

        self.downscale = downscale

        sample = self.observation_space.sample()
        scaled = du.apply(sample, self.scale_image)
        shape = du.apply(scaled, np.shape)

        # make new observation space from scaled

        def make_from_shape(b, shape):
            low, high = b.low.flat[0], b.high.flat[0]
            return gym.spaces.Box(low=low, high=high, shape=shape, dtype=b.dtype)

        space = du.todict(self.observation_space)
        space = du.apply_both(space, shape, make_from_shape)
        self.observation_space = gym.spaces.Dict(space)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        return du.apply(observation, self.scale_image)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return self.observation(observation), reward, terminated, truncated, info

    def scale_image(self, o):
        if not isimg(o):
            return o

        return _scale_image(o, 1 / self.downscale)

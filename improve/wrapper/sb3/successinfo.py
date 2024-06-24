import gymnasium as gym


class SuccessInfoWrapper(gym.Wrapper):
    """ A simple wrapper that adds a is_success key which SB3 tracks"""
    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, terminated, truncated, info

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from tqdm import tqdm


class ProbeEnv(gym.Env):
    def __init__(self):
        super(ProbeEnv, self).__init__()

        # self.observation_space = spaces.Discrete(2)  # Two states: 0 and 1
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Dict({'state': self.observation_space})

        # self.action_space = spaces.Discrete(2)  # Two actions: 0 and 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.current_obs = 0

    def reset(self, seed=None, options=None):
        np.random.seed(seed)

        # self.current_obs = np.random.choice([0, 1])
        obs = {'state': np.random.random() * 2 - 1}
        self.current_obs = obs['state']

        info = {}
        return obs, info

    def step_discrete(self, action):

        if self.current_obs == 0:
            reward = 1 if action == 0 else -1
        else:
            reward = 1 if action == 1 else -1
        self.current_obs = (self.current_obs + 1) % 2

        terminated, truncated = True, True
        info = {}

        return self.current_obs, reward, terminated, truncated, info

    def step(self, action):

        action = np.clip(action, -1, 1)
        reward = 1 - np.abs(self.current_obs - action)  # reward is 1 - distance

        next_obs  = {'state': np.random.random() * 2 - 1}
        terminated,truncated = True, True
        info = {}

        return next_obs, reward, terminated, truncated, info

    def render(self, mode="human", close=False):
        pass


register(id="probe", entry_point="improve.wrapper.probe:ProbeEnv")


def main():

    # Usage example
    env = gym.make("probe")

    for _ in tqdm(range(10)):
        obs, info = env.reset()

        terminated, truncated = False, False
        while not (terminated or truncated):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"State: {obs}, Action: {action}, Reward: {reward}, Next State: {next_obs}"
            )
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            obs = next_obs


if __name__ == "__main__":
    main()

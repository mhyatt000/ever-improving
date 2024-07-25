from gymnasium.core import Wrapper

class AwacRewardWrapper(Wrapper):
    def __init__(self, 
                 env):
        super().__init__(env)
        self.env = env
        
    def step(self, action):
        obs, reward, success, truncated, info = self.env.step(action)
        
        # change reward dist from [0, 1] --> [-1, 0]
        reward = -1 if reward == 0 else 0 if reward == 1 else reward
        return obs, reward, success, truncated, info
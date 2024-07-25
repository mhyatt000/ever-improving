import numpy as np
from gymnasium.core import Wrapper

class StickyGripperWrapper(Wrapper):
    """
    Uses stick gripper for google_robot envs
    """
    def __init__(self, 
                 env,
                 task):
        super().__init__(env)
        self.env = env
        self.task = task
        
        if "widowx_bridge" in self.task:
            self.sticky_gripper_num_repeat = 1
        elif "google_robot" in self.task:
            self.sticky_gripper_num_repeat = 15
            
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        
    def reset(self, **kwargs):
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        
        obs, info = super().reset(**kwargs)
        return obs, info
    
    def step(self, action):
        # print("raw action:", action)
        if "google_robot" in self.task:
            current_gripper_action = action[:, -1] if action.ndim > 1 else action[-1]
        
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            if action.ndim > 1:
                action[:, -1] = relative_gripper_action
            else:
                action[-1] = relative_gripper_action
            
            # print("sticky gripper scaled:", action)
            
        obs, reward, success, truncated, info = self.env.step(action)
        return obs, reward, success, truncated, info
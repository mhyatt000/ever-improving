import gymnasium as gym
import mani_skill2.envs
import simpler_env as simpler
from tqdm import tqdm

if False:
    env = gym.make(
        "PickCube-v0",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        # num_envs=1024,
        obs_mode="state",  # there is also "state_dict", "rgbd", ...
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",  # "rgb_array", # there is also "human", "rgb_array", ...
        renderer_kwargs={
            "offscreen_only": True,
            "device": "cuda:0",
        },
    )

env = simpler.make(
    "google_robot_pick_coke_can",
    obs_mode="state_dict",
    render_mode="rgb_array",  # "rgb_array", # there is also "human", "rgb_array", ...
    # renderer_kwargs={ "offscreen_only": True, "device": "cuda:0", },
)

print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0)  # reset with a seed for determinism
done = False

for _ in tqdm(range(1000000)):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # dones = terminated or truncated

    # env.render()  # a display is required to render
env.close()

_ = [
    "pd_joint_delta_pos",
    "pd_joint_pos",
    "pd_ee_delta_pos",
    "pd_ee_delta_pose",
    "pd_ee_delta_pose_align",
    "pd_joint_target_delta_pos",
    "pd_ee_target_delta_pos",
    "pd_ee_target_delta_pose",
    "pd_joint_vel",
    "pd_joint_pos_vel",
    "pd_joint_delta_pos_vel",
]

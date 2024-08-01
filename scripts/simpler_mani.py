import time
from pprint import pprint

import gymnasium as gym
import hydra
import improve
import matplotlib.pyplot as plt
import mediapy
import mplib
import numpy as np
import sapien.core as sapien
import simpler_env as simpler
from improve.wrapper import dict_util as du
from improve.wrapper import residualrl as rrl
from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from omegaconf import OmegaConf
from omegaconf import OmegaConf as OC
from simpler_env.utils.env.observation_utils import \
    get_image_from_maniskill2_obs_dict
from tqdm import tqdm

"""
env = simpler.make(
    "google_robot_pick_horizontal_coke_can",
    obs_mode="state_dict",  # "rgbd",
    # robot="widowx",
    # sim_freq=513,
    # control_freq=3,
    # control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
    # control_mode="arm_pd_ee_delta_pose_gripper_pd_joint_pos",
    # max_episode_steps=60,
    # scene_name="bridge_table_1_v1",
    # camera_cfgs={"add_segmentation": True},
    # prepackaged_config=False,
    render_mode="human",
    # num_envs=16,
)
"""


def get_image(env, obs):
    """show the right observation for video depending on the robot architecture"""
    image = get_image_from_maniskill2_obs_dict(env, obs)
    return image


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        end_time = time.time()
        self.elapsed = end_time - self.start_time
        print(f"{self.name} took {self.elapsed} seconds to run.")


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result

    return wrapper


"""
def setup_planner():
    robot = env.agent.robot
    link_names = [link.get_name() for link in robot.get_links()]
    joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    self.planner = mplib.Planner(
        urdf=urdf_path,
        # srdf="../assets/robot/panda/panda.srdf",
        user_link_names=link_names,
        user_joint_names=joint_names,
        move_group=eef,
        joint_vel_limits=np.ones(7),
        joint_acc_limits=np.ones(7),
    )
    return planner


planner = setup_planner()
planner.IK()
target = tcp.pose.p
print(type(target))
"""


def _rescale_action_with_bound(
    actions: np.ndarray,
    low: float,
    high: float,
    safety_margin: float = 0.0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> np.ndarray:
    """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
    resc_actions = (actions - low) / (high - low) * (
        post_scaling_max - post_scaling_min
    ) + post_scaling_min
    return np.clip(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )


def _unnormalize_action_widowx_bridge(
    action: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    action["world_vector"] = _rescale_action_with_bound(
        action["world_vector"],
        low=-1.75,
        high=1.75,
        post_scaling_max=0.05,
        post_scaling_min=-0.05,
    )
    action["rotation_delta"] = _rescale_action_with_bound(
        action["rotation_delta"],
        low=-1.4,
        high=1.4,
        post_scaling_max=0.25,
        post_scaling_min=-0.25,
    )
    return action


def preprocess_gripper(action: np.ndarray) -> np.ndarray:
    # filter small actions to smooth
    action = np.where(np.abs(action) < 1e-3, 0, action)
    # binarize gripper
    action = 2.0 * (action > 0.0) - 1.0
    return action


def preprocess_action(action: np.ndarray) -> np.ndarray:
    action = {
        "world_vector": action[:3],
        "rotation_delta": action[3:6],
        "gripper": action[-1],
    }
    action["gripper"] = preprocess_gripper(action["gripper"])
    action = _unnormalize_action_widowx_bridge(action)
    action = np.concatenate(
        [
            action["world_vector"],
            action["rotation_delta"],
            np.array([action["gripper"]]),
        ]
    )
    return action


# keep the eef pointed down
qpos = np.array(
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


class Controller:

    def __init__(self):
        self.speed = 0.90
        self.dist = 0.01

        self.open = 0
        self.tolift = False

    def __call__(self, obs):
        obs = du.todict(obs)

        wrt = obs["extra_tcp_to_obj_pos"]
        x, y, z = wrt

        if self.tolift:
            return np.array([0, 0, 0.1, 0, 0, 0, self.speed])

        if abs(x) > self.dist and abs(y) > self.dist:
            xdir, ydir = np.sign(x), np.sign(y)
            return np.array(
                [
                    -xdir * self.speed * abs(x),
                    -ydir * self.speed * abs(y),
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
        if abs(z) > self.dist:
            zdir = np.sign(z)
            return np.array([0, 0, zdir * self.speed * abs(z), 0, 0, 0, 0])
        if self.open < 8:
            self.open += 1
            return np.array([0, 0, 0, 0, 0, 0, self.speed])

        self.tolift = True

        return np.array([0, 0, -0.01, 0, 0, 0, self.speed])

    def reset(self):
        self.open = 0
        self.tolift = False


d = {
    "agent": {
        "base_pose": (7,),
        "controller": {"gripper": {"target_qpos": (2,)}},
        "qpos": (11,),
        "qvel": (11,),
    },
    "extra": {"obj_pose": (7,), "tcp_pose": (7,), "tcp_to_obj_pos": (3,)},
}

"""
dd = {
    "agent": {
        "controller": {"gripper": {"target_qpos": array([0.0, 0.0], dtype=float32)}},
        "qpos": array( [ -0.26394573, 0.08319134, 0.50176114, 1.156859, 0.02858367, 1.5925982, -1.080653, 0.0, 0.0, -0.00285961, 0.7851361, ],),
        "qvel": array( [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float32),
    },
    "extra": {
        "obj_pose": array(
            [ -0.22294247, 0.302678, 0.9202228, 0.00230632, -0.29257464, 0.00500705, 0.9562269, ], dtype=float32,),
        "tcp_pose": array(
            [ -0.08938442, 0.41841692, 1.0409188, 0.0324987, -0.7801727, -0.6038523, 0.16011547, ],
            dtype=float32,
        ),
        "tcp_to_obj_pos": array([-0.13355805, -0.11573893, -0.12069601], dtype=float32),
    },
}
"""


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):

    assert cfg.get("sweep_id", None) == "debug"
    assert (
        cfg.env.foundation.name is None
    ), f"cfg.env.foundation.name is not None: {cfg.env.foundation.name}"

    pprint(OmegaConf.to_container(cfg, resolve=True))

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # env = simpler.make( "google_robot_pick_horizontal_coke_can", obs_mode="state_dict", max_episode_steps=60, render_mode="human",)

    # render_mode = "human"
    render_mode = "cameras"
    env = rrl.make(
        cfg.env,
        # "google_robot_pick_horizontal_coke_can",
        obs_mode="state_dict",  # "rgbd",
        # robot="widowx",
        # sim_freq=513,
        # control_freq=3,
        # control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        control_mode="tmp",
        # max_episode_steps=60,
        # scene_name="bridge_table_1_v1",
        # camera_cfgs={"add_segmentation": True},
        prepackaged_config=False,
        render_mode=render_mode,
        # num_envs=16,
    )

    mode = [
        "arm_pd_ee_delta_pose_gripper_pd_joint_delta_pos",
        "arm_pd_ee_delta_pose_gripper_pd_joint_target_delta_pos", # use?
        "arm_pd_ee_delta_pose_align_gripper_pd_joint_delta_pos",
        "arm_pd_ee_delta_pose_align_gripper_pd_joint_target_delta_pos",
    ]

    print("ok")
    quit()

    eef = env.agent.config.ee_link_name
    tcp = get_entity_by_name(env.agent.robot.get_links(), eef)

    # observation, reward, success, truncated, info = env.step(zeros)

    C = Controller()
    img_arr = []

    for _ in tqdm(range(3), leave=False):
        obs, info = env.reset()
        env.agent.reset(qpos)
        C.reset()

        # action = env.action_space.sample()
        # action = np.zeros_like(action)

        done = False
        steps = 0
        T = Timer("loop")
        with T:
            while not done:

                # action = C(obs)
                action = env.action_space.sample()
                action = preprocess_action(action)

                # print(action)

                # print(action)
                obs, reward, success, terminated, info = env.step(action)
                done = terminated or success
                steps += 1

                if render_mode == "cameras":
                    imgs = env.render()
                    # select the right image
                    # total is 512,3200,3
                    # select the second image
                    # imgs = imgs[:,512:1024,:]
                    imgs = imgs[:, :1024, :]
                    img_arr.append(imgs)
                else:
                    env.render()

                # tcp = get_entity_by_name(env.env.agent.robot.get_links(), eef)
                # print(list(obs["agent"]["qpos"]))

                # print(obs["agent"]["qpos"][-4:-2])
                # print(img.shape)
                # plt.imshow(imgs)
                # plt.pause(0.01)

        print(f"completed in {steps} steps")
        print(round(T.elapsed / 60), 6)

    if render_mode == "cameras":
        mediapy.write_video("controller.gif", img_arr, fps=20, codec="gif")

        # time.sleep(3)
        # quit()


if __name__ == "__main__":
    main()

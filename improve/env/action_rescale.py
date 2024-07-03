"""
functions for rescaling the action space
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


class ActionRescaler:

    def __init__(self, strategy, residual_scale):
        assert strategy in ["dynamic", "clip", None]
        self.strategy = strategy
        self.residual_scale = residual_scale

        assert strategy == "dynamic"  # only dynamic is debugged for central FM

        if strategy == "clip":
            translation = np.linalg.norm([0.05, 0.05, 0.05])
            axis, angle = self.rpy_to_axis_angle(*[0.25, 0.25, 0.25])
            self.max = {"translation": translation, "rotation": angle}

        if strategy == "dynamic":
            self.bounds = [
                (0.05, -0.05),
                (0.05, -0.05),
                (0.05, -0.05),
                (0.25, -0.25),
                (0.25, -0.25),
                (0.25, -0.25),
                (1, -1),
            ]

    def pad_act_for_fm(self, action):
        """Pad the RP action space to match the FM action space."""
        # action is (1,3)
        bs, seq = action.shape
        padlen = 7 - seq
        return np.concatenate([action, np.zeros((bs, padlen))], axis=1)

    def compute_final_action(self, action, model_action):

        action = self.pad_act_for_fm(action)

        # actions are added together using the rp_scale
        # if the action is out of bounds, it is transformed to be in bounds
        if self.strategy == "clip":
            total_action = model_action + (action * self.residual_scale)
            translation = np.linalg.norm(total_action[:3])
            axis, rotation = self.rpy_to_axis_angle(*total_action[3:6])

            # dont go out of bounds
            if abs(translation) > self.max["translation"]:
                print("OOB translation", total_action[:3])
                total_action[:3] = total_action[:3] * (
                    self.max["translation"] / translation
                )
                print(total_action[:3])
            if abs(rotation) > self.max["rotation"]:
                print("OOB rotation", total_action[3:6])
                total_action[3:6] = self.axis_angle_to_rpy(axis, self.max["rotation"])
                print(total_action[3:6])

            return total_action

        # residual actions transformed to the remaining action space after FM
        # added together without rp_scale
        if self.strategy == "dynamic":
            bounds = [
                (high - a, low - a) for (high, low), a in zip(self.bounds, model_action)
            ]

            def f(x, b):
                return asymmetric_transform(
                    x,
                    low=-1,
                    high=1,
                    post_scaling_max=b[0],
                    post_scaling_min=b[1],
                )

            action = np.array([f(a, b) for a, b in zip(action, bounds)])
            return action + model_action

    def scale_action(self, action):
        return self.dict2act(_scale_for_obs(self.act2dict(action)))

    def unscale_for_obs(self, action):
        return self.dict2act(_unscale_for_obs(self.act2dict(action)))

    def dict2act(self, action: dict[str, np.ndarray]) -> np.ndarray:
        print(action)
        return np.concatenate(
            [
                action["world_vector"],
                action["rot_axangle"],
                np.array([action["gripper"]]),
            ],
            axis=-1,
        )

    def act2dict(self, action: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "world_vector": action[:, :3],
            "rot_axangle": action[:, 3:6],
            "gripper": action[:, -1],
        }


def _unscale_for_obs(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    action["world_vector"] = _rescale_action_with_bound(
        action["world_vector"],
        low=-0.05,
        high=0.05,
        post_scaling_max=1.75,
        post_scaling_min=-1.75,
    )
    action["rot_axangle"] = _rescale_action_with_bound(
        action["rot_axangle"],
        low=-0.25,
        high=0.25,
        post_scaling_max=1.4,
        post_scaling_min=-1.4,
    )
    return action


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


def asymmetric_transform(
    actions: np.ndarray,
    low: float,
    high: float,
    safety_margin: float = 0.0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> np.ndarray:
    """Modified rescale function ensuring 0 maps to 0."""

    # dont divide by zero
    pos_scale = post_scaling_max / max(high, 1e-8)
    neg_scale = post_scaling_min / min(low, -1e-8)

    resc_actions = np.where(actions >= 0, actions * pos_scale, actions * neg_scale)
    return np.clip(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )

"""
functions for rescaling the action space
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from improve import cn


class ActionRescaler:

    def __init__(self, strategy=cn.Strategy.CLIP, residual_scale=1.0):
        self.strategy = strategy.value
        self.residual_scale = residual_scale

        if self.strategy == "clip":
            translation = np.linalg.norm([0.05, 0.05, 0.05])
            axis, angle = rpy_to_axis_angle(*[0.25, 0.25, 0.25])
            self.max = {"translation": translation, "rotation": angle}

        if self.strategy == "dynamic":
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
            action = self.scale_action(action)  # to RTX space

            total_action = model_action + (action * self.residual_scale)

            # vectorized now
            translation = np.linalg.norm(total_action[:, :3], axis=-1)
            rpy = total_action[:, 3:6]

            results = [rpy_to_axis_angle(*item) for item in rpy]
            axis = np.array([r[0] for r in results])
            rotation = np.array([r[1] for r in results])

            for i in range(len(total_action)):
                if abs(translation[i]) > self.max["translation"]:
                    total_action[i, :3] = total_action[i, :3] * (
                        self.max["translation"] / translation[i]
                    )

                if abs(rotation[i]) > self.max["rotation"]:
                    total_action[i, 3:6] = axis_angle_to_rpy(
                        axis[i], self.max["rotation"]
                    )

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
        
        ### CHANGED
        # if self.strategy == "awac_rescale":
        #     return self.dict2act(_scale_action_awac(self.act2dict(action)))
            
        if self.strategy is None:
            action = self.scale_action(action)  # to RTX space
            return action + model_action

    def scale_action(self, action):
        return self.dict2act(_scale_action(self.act2dict(action)))

    def unscale_for_obs(self, action):
        return self.dict2act(_unscale_for_obs(self.act2dict(action)))

    def dict2act(self, action: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                action["world_vector"],
                action["rot_axangle"],
                action["gripper"],
            ],
            axis=-1,
        )

    def act2dict(self, action: np.ndarray) -> dict[str, np.ndarray]:
        try:
            return {
                "world_vector": action[:, :3],
                "rot_axangle": action[:, 3:6],
                "gripper": np.expand_dims(action[:, -1], axis=-1),
            }
        except:
            return {
                "world_vector": action[:, :3],
                "rot_axangle": action[:, 3:6],
                "gripper": action[:, -1],
            }

### CHANGED (for awac rescaling)
# def _scale_action_awac(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
#     action["world_vector"] = _rescale_action_with_bound(
#         action["world_vector"],
#         low=-1.75,
#         high=1.75,
#         post_scaling_min=-1,
#         post_scaling_max=1,
#     )

#     action["rot_axangle"] = _rescale_action_with_bound(
#         action["rot_axangle"],
#         low=-1.4,
#         high=1.4,
#         post_scaling_min=-1,
#         post_scaling_max=1,
#     )
#     return action

def _scale_action(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    action["world_vector"] = _rescale_action_with_bound(
        action["world_vector"],
        low=-1,    #-1.75,
        high=1,    #1.75,
        post_scaling_min=-0.05,
        post_scaling_max=0.05,
    )

    action["rot_axangle"] = _rescale_action_with_bound(
        action["rot_axangle"],
        low=-1,        #-1.4,
        high=1,       #1.4,
        post_scaling_min=-0.25,
        post_scaling_max=0.25,
    )
    return action


def _unscale_for_obs(action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    action["world_vector"] = _rescale_action_with_bound(
        action["world_vector"],
        low=-0.05,
        high=0.05,
        post_scaling_min=-1,   #-1.75,
        post_scaling_max=1,   #1.75,
    )
    action["rot_axangle"] = _rescale_action_with_bound(
        action["rot_axangle"],
        low=-0.25,
        high=0.25,
        post_scaling_min=-1,   #-1.4,
        post_scaling_max=1,   #1.4,
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

    """ do not use this 
    """
    return np.clip(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )
    return resc_actions


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


def rpy_to_axis_angle(roll, pitch, yaw):

    rotation = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
    axis_angle = rotation.as_rotvec()

    # The angle is the magnitude of the rotation vector
    angle = np.linalg.norm(axis_angle)

    # The axis is the normalized rotation vector
    # This should be [0, 0, 0] if there is no rotation
    axis = axis_angle / angle if angle != 0 else axis_angle

    return axis, angle


def axis_angle_to_rpy(axis, angle):
    rotation = R.from_rotvec(axis * angle)
    rpy = rotation.as_euler("xyz", degrees=False)
    return rpy


def main():

    rescaler = ActionRescaler("dynamic", 1)

    fm_action = np.array([[0.1, -0.1, 0, 0.1, -0.1, 0, 1]])
    fm_action = np.zeros((1, 7))

    print(rescaler.scale_action(fm_action))


if __name__ == "__main__":
    main()

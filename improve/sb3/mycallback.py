from pprint import pprint
from typing import (Any, Dict, List, Mapping, Optional, Sequence, TextIO,
                    Tuple, Union)

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, Logger

import wandb

""" for EvalCallback

- eval_env: The environment used for initialization
- callback_on_new_best: Callback to trigger
    when there is a new best model according to the ``mean_reward``
- callback_after_eval: Callback to trigger after every evaluation
- n_eval_episodes: The number of episodes to test the agent
- eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
- log_path: Path to a folder where the evaluations (``evaluations.npz``)
    will be saved. It will be updated at each evaluation.
- render: Whether to render or not the environment during evaluation
- verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
"""


class PlottingCallback(BaseCallback):
    """
    Callback for plotting the performance in realtime.

    :param verbose: (int)
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._plot = None

    def _on_step(self) -> bool:
        # get the monitor's data
        x, y = ts2xy(load_results(log_dir), "timesteps")
        if self._plot is None:  # make the plot
            plt.ion()
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            (line,) = ax.plot(x, y)
            self._plot = (line, ax, fig)
            plt.show()
        else:  # update and rescale the plot
            self._plot[0].set_data(x, y)
            self._plot[-2].relim()
            self._plot[-2].set_xlim(
                [
                    self.locals["total_timesteps"] * -0.02,
                    self.locals["total_timesteps"] * 1.02,
                ]
            )
            self._plot[-2].autoscale_view(True, True, True)
            self._plot[-1].canvas.draw()


class WandbLogger(Logger):

    def __init__(self, folder: Optional[str], output_formats: List[KVWriter]):
        super().__init__(folder, output_formats)
        self.key_count = {}

    def dump(self, step: int = 0) -> None:
        """ Write all of the diagnostics from the current iteration """
        wandb.log(self.name_to_value, step=step)
        super().dump(step)


class MyCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

        # self.model = None  # type: BaseRLModel
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # self.logger = None  # type: logger.Logger
        # self.parent = None  # type: Optional[BaseCallback]

        self.is_training = False
        self.previous_num_timesteps = 0
        self.n_updates = 0
        self.last_update = 0

    def _on_training_start(self) -> None:
        """This method is called before the first rollout starts."""
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        print("rollout")
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        info = to_nest(self.logger.name_to_value)
        # count = to_nest(self.logger.name_to_count)
        # exclude = to_nest(self.logger.name_to_excluded)

        self.n_updates = info.get("train", {}).get("n_updates", self.n_updates)

        if self.n_updates > self.last_update:
            print(info)
            wandb.log(info)  #  step=i)

        self.last_update = self.n_updates
        return True

    def _on_rollout_end(self) -> None:
        """This event is triggered before updating the policy."""

        info = to_nest(self.logger.name_to_value)
        print(info)
        return True

        self.n_updates = info.get("train", {}).get("n_updates", self.n_updates)

        if self.n_updates > self.last_update:
            print(info)
            wandb.log(info)  #  step=i)

        self.last_update = self.n_updates
        return True

    def _on_training_end(self) -> None:
        """This event is triggered before exiting the `learn()` method."""
        pass

    def on_eval_step(self) -> None:
        """This event is triggered before each evaluation rollout
        ** custom
        """
        pass


def to_nest(d):
    result = {}
    for key, value in d.items():
        parts = key.split("/")
        current_level = result
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        current_level[parts[-1]] = value
    return result


"""
locals = {
    "actions": array(
        [
            [ 0.00575342, 1.2458076, 1.0823157, -0.3306231, 0.29876664, 0.80246663, 1.4914905, ]
        ],
        dtype=float32,
    ),
    "callback": CallbackList,
    "clipped_actions": array(
        [[0.00575342, 1.0, 1.0, -0.3306231, 0.29876664, 0.80246663, 1.0]], dtype=float32
    ),
    "done": False,
    "dones": array([False]),
    "env": VecTransposeImage,
    "idx": 0,
    "infos": [
        {
            "TimeLimit.truncated": False,
            "consecutive_grasp": False,
            "elapsed_steps": 1,
            "episode_stats": OrderedDict(
                [
                    ("moved_correct_obj", False),
                    ("moved_wrong_obj", False),
                    ("is_src_obj_grasped", False),
                    ("consecutive_grasp", False),
                    ("src_on_target", False),
                ]
            ),
            "is_src_obj_grasped": False,
            "moved_correct_obj": False,
            "moved_wrong_obj": False,
            "src_on_target": False,
            "success": False,
        }
    ],
    "iteration": 0,
    "log_interval": 1,
    "log_probs": tensor([-9.3278], device="cuda:0"),
    "n_rollout_steps": 16,
    "n_steps": 12,
    "new_obs": OrderedDict(
        [
            (
                "image",
                array(
                    [
                        [
                            [
                                [102, 102, 102, 23, 16, 20],
                                [102, 102, 102, 20, 20, 23],
                                [102, 102, 102, 20, 23, 28],
                                [16, 23, 23, 116, 112, 92],
                                [16, 23, 23, 112, 92, 92],
                                [16, 23, 23, 116, 95, 92],
                            ],
                            [
                                [88, 88, 88, 21, 17, 21],
                                [88, 88, 88, 21, 21, 21],
                                [88, 88, 88, 21, 21, 24],
                                [17, 21, 21, 80, 83, 73],
                                [17, 21, 21, 83, 73, 73],
                                [17, 21, 21, 80, 82, 73],
                            ],
                            [
                                [81, 81, 81, 22, 23, 23],
                                [81, 81, 81, 23, 23, 22],
                                [81, 81, 81, 23, 22, 24],
                                [23, 22, 22, 41, 53, 58],
                                [23, 22, 22, 53, 58, 58],
                                [23, 22, 22, 41, 70, 58],
                            ],
                        ]
                    ],
                    dtype=uint8,
                ),
            ),
            (
                "partial_action",
                array(
                    [
                        [
                            -0.00122557,
                            -0.00149062,
                            -0.00660526,
                            0.00162785,
                            0.02118792,
                            0.007247,
                            1.0,
                        ]
                    ],
                    dtype=float32,
                ),
            ),
        ]
    ),
    "obs_tensor": {
        "image": tensor(
            [
                [
                    [
                        [102, 102, 102, 23, 16, 20],
                        [102, 102, 102, 20, 20, 23],
                        [102, 102, 102, 20, 23, 28],
                        [16, 23, 23, 116, 112, 92],
                        [16, 23, 23, 112, 92, 92],
                        [16, 23, 23, 116, 95, 92],
                    ],
                    [
                        [88, 88, 88, 21, 17, 21],
                        [88, 88, 88, 21, 21, 21],
                        [88, 88, 88, 21, 21, 24],
                        [17, 21, 21, 80, 83, 73],
                        [17, 21, 21, 83, 73, 73],
                        [17, 21, 21, 80, 82, 73],
                    ],
                    [
                        [81, 81, 81, 22, 23, 23],
                        [81, 81, 81, 23, 23, 22],
                        [81, 81, 81, 23, 22, 24],
                        [23, 22, 22, 41, 53, 58],
                        [23, 22, 22, 53, 58, 58],
                        [23, 22, 22, 41, 70, 58],
                    ],
                ]
            ],
            device="cuda:0",
            dtype=torch.uint8,
        ),
        "partial_action": tensor(
            [[-0.0084, -0.0077, -0.0137, -0.0323, 0.0305, 0.0971, 1.0000]],
            device="cuda:0",
        ),
    },
    "progress_bar": True,
    "reset_num_timesteps": True,
    "rewards": array([0.0], dtype=float32),
    "rollout_buffer": DictRolloutBuffer,
    "self": PPO,
    "tb_log_name": "PPO",
    "total_timesteps": 10000,
    "values": tensor([[0.0]], device="cuda:0"),
}
"""


def main():
    pass

if __name__ == "__main__":
    main()

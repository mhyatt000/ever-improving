import copy
from pprint import pprint

import hydra
import improve
import improve.hydra.resolver
import jax
import jax.numpy as jnp
import numpy as np
import simpler_env as simpler
import tensorflow as tf
import tensorflow_hub as hub
from flax.training import checkpoints
from improve.oxe_rt.rt1_model import RT1, detokenize_action
from improve.wrapper import dict_util as du
from stable_baselines3.common.vec_env import SubprocVecEnv
from transforms3d.euler import euler2axangle


class RT1Policy:
    """Runs inference with a RT-1 policy."""

    def __init__(
        self,
        checkpoint_path=None,
        model=RT1(),
        variables=None,
        seqlen=15,
        batch_size=1,
        action_scale=1,
        rng=None,
    ):
        """Initializes the policy.

        Args:
          checkpoint_path: A checkpoint point from which to load variables. Either
            this or variables must be provided.
          model: A nn.Module to use for the policy. Must match with the variables
            provided by checkpoint_path or variables.
          variables: If provided, will use variables instead of loading from
            checkpoint_path.
          seqlen: The history length to use for observations.
          rng: a jax.random.PRNGKey to use for the random number generator.
        """
        if not variables and not checkpoint_path:
            raise ValueError(
                "At least one of `variables` or `checkpoint_path` must be defined."
            )
        self.model = model
        self._checkpoint_path = checkpoint_path
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.action_scale = action_scale

        self.unnormalize_action = False
        self.unnormalize_action_fxn = None
        self.invert_gripper_action = False
        self.action_rotation_mode = "axis_angle"

        self._run_action_inference_jit = jax.jit(self._run_action_inference)
        # for debugging
        # self._run_action_inference_jit = self._run_action_inference

        if rng is None:
            self.rng = jax.random.PRNGKey(0)
        else:
            self.rng = rng

        if variables:
            self.variables = variables
        else:
            state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)
            variables = {
                "params": state_dict["params"],
                "batch_stats": state_dict["batch_stats"],
            }
            print("Loaded variables from checkpoint.")
            # print('params', variables['params'])
            # print('batch_stats', variables['batch_stats'])
            self.variables = variables

    def _small_action_filter_google_robot(
        self,
        raw_action: dict[str, np.ndarray | tf.Tensor],
        arm_movement: bool = False,
        gripper: bool = True,
    ) -> dict[str, np.ndarray | tf.Tensor]:
        # small action filtering for google robot
        if arm_movement:
            raw_action["world_vector"] = tf.where(
                tf.abs(raw_action["world_vector"]) < 5e-3,
                tf.zeros_like(raw_action["world_vector"]),
                raw_action["world_vector"],
            )
            raw_action["rotation_delta"] = tf.where(
                tf.abs(raw_action["rotation_delta"]) < 5e-3,
                tf.zeros_like(raw_action["rotation_delta"]),
                raw_action["rotation_delta"],
            )
            raw_action["base_displacement_vector"] = tf.where(
                raw_action["base_displacement_vector"] < 5e-3,
                tf.zeros_like(raw_action["base_displacement_vector"]),
                raw_action["base_displacement_vector"],
            )
            raw_action["base_displacement_vertical_rotation"] = tf.where(
                raw_action["base_displacement_vertical_rotation"] < 1e-2,
                tf.zeros_like(raw_action["base_displacement_vertical_rotation"]),
                raw_action["base_displacement_vertical_rotation"],
            )
        if gripper:
            raw_action["gripper_closedness_action"] = tf.where(
                tf.abs(raw_action["gripper_closedness_action"]) < 1e-2,
                tf.zeros_like(raw_action["gripper_closedness_action"]),
                raw_action["gripper_closedness_action"],
            )
        return raw_action

    def _run_action_inference(self, observation, rng):
        """A jittable function for running inference."""

        # We add zero action tokens so that the shape is (seqlen, 11).
        # Note that in the vanilla RT-1 setup, where
        # `include_prev_timesteps_actions=False`, the network will not use the
        # input tokens and instead uses zero action tokens, thereby not using the
        # action history. We still pass it in for simplicity.
        act_tokens = jnp.zeros((self.batch_size, 6, 11))

        # Add a batch dim to the observation.
        # batch_obs = jax.tree_map(lambda x: jnp.expand_dims(x, 0), observation)

        _, random_rng = jax.random.split(rng)

        # act = {
        #   "world_vector": jnp.ones((self.batch_size, 15, 3)),
        #   "rotation_delta": jnp.ones((self.batch_size, 15, 3)),
        #   "gripper_closedness_action": jnp.ones((self.batch_size, 15, 1)),
        #   "base_displacement_vertical_rotation": jnp.ones((self.batch_size, 15, 1)),
        #   "base_displacement_vector": jnp.ones((self.batch_size, 15, 2)),
        #   "terminate_episode": jnp.ones((self.batch_size, 15, 3), dtype=jnp.int32),
        # }

        # pprint(du.apply(observation, lambda x: x.shape))

        output_logits = self.model.apply(
            self.variables,
            observation,
            act=None,
            act_tokens=act_tokens,
            train=False,
            rngs={"random": random_rng},
        )

        time_step_tokens = self.model.num_image_tokens + self.model.num_action_tokens
        output_logits = jnp.reshape(
            output_logits, (self.batch_size, self.seqlen, time_step_tokens, -1)
        )
        action_logits = output_logits[:, -1, ...]
        action_logits = action_logits[:, self.model.num_image_tokens - 1 : -1]

        action_logp = jax.nn.softmax(action_logits)
        action_token = jnp.argmax(action_logp, axis=-1)

        # Detokenize the full action sequence.
        detokenized = detokenize_action(
            action_token, self.model.vocab_size, self.model.world_vector_range
        )

        # if self.batch_size == 1:
        # detokenized = jax.tree_map(lambda x: x[0], detokenized)

        return detokenized

    def _add_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        images = np.stack(self.image_history, axis=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(
            horizon, dtype=np.float64
        )  # note: this should be of float type, not a bool type
        pad_mask[: horizon - min(horizon, self.num_image_history)] = 0
        # pad_mask = np.ones(self.horizon, dtype=np.float64) # note: this should be of float type, not a bool type
        # pad_mask[:self.horizon - self.num_image_history] = 0
        return images, pad_mask

    def reset(self, task_description: str) -> None:
        self.image_history.clear()
        self.num_image_history = 0

    def action(self, observation):
        """Outputs the action given observation from the env."""

        # Assume obs has no batch dimensions.
        observation = copy.deepcopy(observation)

        # Jax does not support string types, so remove it from the dict if it
        # exists.
        if "natural_language_instruction" in observation:
            del observation["natural_language_instruction"]

        # Resize using TF image resize to avoid any issues with using different
        # resize implementation, since we also use tf.image.resize in the data
        # pipeline. Also scale image to [0, 1].

        images = []
        for image in observation["image"]:
            image = tf.image.resize(image, (300, 300)).numpy()
            image /= 255.0
            images.append(image)
        observation["image"] = np.stack(images)

        print('pause here')
        quit()

        self._add_to_history(obs)
        obs, pad_mask = self._obtain_image_history_and_mask()
        # i think this is for batch? idk
        # obs, pad_mask = obs[None], pad_mask[None]



        self.rng, rng = jax.random.split(self.rng)
        action = self._run_action_inference_jit(observation, rng)
        action = jax.device_get(action)

        """
        # Use the base pose mode if the episode if the network outputs an invalid
        # `terminate_episode` action.
        if np.sum(action["terminate_episode"]) == 0:
            action["terminate_episode"] = np.zeros_like(action["terminate_episode"])
            action["terminate_episode"][-1] = 1
        """

        raw_action = action

        # do the postprocessing itemwise without rewriting everything for now
        a = [du.apply(raw_action, lambda x: x[i]) for i in range(self.batch_size)]
        b = [du.apply(action, lambda x: x[i]) for i in range(self.batch_size)]
        raw_action, action = zip(
            *[self.simpler_postprocessing(_a, _b) for _a, _b in zip(a, b)]
        )
        raw_action, action = du.stack(raw_action), du.stack(action)

        print("raw_action", raw_action)
        print("action", action)
        quit()
        return raw_action, action

    def simpler_postprocessing(self, raw_action, action):

        if self.policy_setup == "google_robot":
            raw_action = self._small_action_filter_google_robot(
                raw_action, arm_movement=False, gripper=True
            )
        if self.unnormalize_action:
            raw_action = self.unnormalize_action_fxn(raw_action)
        for k in raw_action.keys():
            raw_action[k] = np.asarray(raw_action[k])

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = (
            np.asarray(raw_action["world_vector"], dtype=np.float64) * self.action_scale
        )
        if self.action_rotation_mode == "axis_angle":
            action_rotation_delta = np.asarray(
                raw_action["rotation_delta"], dtype=np.float64
            )
            action_rotation_angle = np.linalg.norm(action_rotation_delta)
            action_rotation_ax = (
                action_rotation_delta / action_rotation_angle
                if action_rotation_angle > 1e-6
                else np.array([0.0, 1.0, 0.0])
            )

            action["rot_axangle"] = (
                action_rotation_ax * action_rotation_angle * self.action_scale
            )
        elif self.action_rotation_mode in ["rpy", "ypr", "pry"]:
            if self.action_rotation_mode == "rpy":
                roll, pitch, yaw = np.asarray(
                    raw_action["rotation_delta"], dtype=np.float64
                )
            elif self.action_rotation_mode == "ypr":
                yaw, pitch, roll = np.asarray(
                    raw_action["rotation_delta"], dtype=np.float64
                )
            elif self.action_rotation_mode == "pry":
                pitch, roll, yaw = np.asarray(
                    raw_action["rotation_delta"], dtype=np.float64
                )
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action["rot_axangle"] = (
                action_rotation_ax * action_rotation_angle * self.action_scale
            )
        else:
            raise NotImplementedError()

        raw_gripper_closedness = raw_action["gripper_closedness_action"]
        if self.invert_gripper_action:
            # rt1 policy output is uniformized such that -1 is open gripper, 1 is close gripper;
            # thus we need to invert the rt1 output gripper action for some embodiments like WidowX, since for these embodiments -1 is close gripper, 1 is open gripper
            raw_gripper_closedness = -raw_gripper_closedness
        if self.policy_setup == "google_robot":
            # gripper controller: pd_joint_target_delta_pos_interpolate_by_planner; raw_gripper_closedness has range of [-1, 1]
            action["gripper"] = np.asarray(raw_gripper_closedness, dtype=np.float64)
        elif self.policy_setup == "widowx_bridge":
            # gripper controller: pd_joint_pos; raw_gripper_closedness has range of [-1, 1]
            action["gripper"] = np.asarray(raw_gripper_closedness, dtype=np.float64)
            # binarize gripper action to be -1 or 1
            action["gripper"] = 2.0 * (action["gripper"] > 0.0) - 1.0
        else:
            raise NotImplementedError()

        action["terminate_episode"] = raw_action["terminate_episode"]

        # update policy state
        # self.policy_state = policy_step.state

        return raw_action, action


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):

    import os.path as osp

    import numpy as np
    import wandb
    from omegaconf import OmegaConf as OC

    BATCH_SIZE = cfg.env.n_envs

    sequence_length = 15
    num_action_tokens = 11
    layer_size = 256
    vocab_size = 512
    num_image_tokens = 81
    rt1x_model = RT1(
        num_image_tokens=num_image_tokens,
        num_action_tokens=num_action_tokens,
        layer_size=layer_size,
        vocab_size=vocab_size,
        # Use token learner to reduce tokens per image to 81.
        use_token_learner=True,
        # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
        world_vector_range=(-2.0, 2.0),
    )
    policy = RT1Policy(
        checkpoint_path="/home/zero-shot/rt_1_x_jax/b321733791_75882326_000900000",
        model=rt1x_model,
        seqlen=sequence_length,
        batch_size=BATCH_SIZE,
    )

    # print(policy.model.state_dict())
    # obs = {
    #     'image': jnp.ones((2, 15, 300, 300, 3)),
    #     'natural_language_embedding': jnp.ones((2, 15, 512)),
    # }

    # print(policy.action(obs))

    print(rt1x_model)
    print(policy)

    from improve.env import make_env, make_envs

    if cfg.job.wandb.use:
        print("Using wandb")
        run = wandb.init(
            project="residualrl-maniskill2demo",
            dir=cfg.callback.log_path,
            job_type="train",
            # sync_tensorboard=True,
            monitor_gym=True,
            name=cfg.job.wandb.name,
            group=cfg.job.wandb.group,
            config=OC.to_container(cfg, resolve=True),
        )
        wandb.config.update({"name": run.name})

    pprint(OC.to_container(cfg, resolve=True))  # keep after wandb so it logs

    log_dir = osp.join(cfg.callback.log_path, wandb.run.name)
    eval_only = not cfg.train.use_train
    num_envs = cfg.env.n_envs
    max_episode_steps = cfg.env.max_episode_steps

    env, eval_env = make_envs(
        cfg,
        log_dir,
        eval_only=eval_only,
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
    )
    print(env)

    #
    #
    #

    lang_embed_model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    )
    instructions = env.env_method("get_language_instruction")

    # create 4 vectorized environments
    # env = SubprocVecEnv([make_env(cfg) for _ in range(BATCH_SIZE)])
    # env = SubprocVecEnv( [lambda: simpler.make(cfg.env.foundation.task) for _ in range(BATCH_SIZE)])
    # env.seed(cfg.env.seed)
    obs = env.reset()

    def add_seq_dim_padding(data, sequence_length=15):
        batch, data_shape = data.shape[0], data.shape[1:]
        mask = np.zeros((batch, sequence_length, *data_shape))

        if len(data.shape) == 4:
            mask[:, 0, :, :] = data
        else:
            mask[:, 0, :] = data

        return mask

    processed_obs = {
        "image": None,
        "natural_language_embedding": add_seq_dim_padding(
            (lang_embed_model(instructions)).numpy()
        ),
    }

    # dummy = { "image": jnp.ones((BATCH_SIZE, 15, 300, 300, 3)), "natural_language_embedding": jnp.ones((BATCH_SIZE, 15, 512)), }

    images = []
    rewards, dones = 0, 0
    while True:
        obs = obs["simpler-img"]  # [None]
        obs = obs

        untranspose = lambda x: np.transpose(x, (0, 2, 3, 1))
        # obs = untranspose(obs)

        # print(obs.shape)
        # print(obs[None].shape)

        processed_obs["image"] = add_seq_dim_padding(obs)
        images.append(obs)

        # print(images[0].shape)
        # print(len(images))

        # raw_action, action = policy.action(dummy)
        raw_action, action = policy.action(processed_obs)
        action = np.concatenate(
            [action["world_vector"], action["rot_axangle"], action["gripper"]], axis=-1
        )

        print(action)
        quit()

        obs, reward, done, _ = env.step(action)
        # print("same", np.all(obs['simpler-img'][0] == obs['simpler-img'][1]), obs['simpler-img'][0].shape)
        print("raw_action", raw_action)
        print("action", action)

        # if np.all(prev_obs == obs['simpler-img']):
        # print("ENV IS NOT STEPPING")

        rewards += float(sum(reward))
        dones += float(sum(done))

        print(
            f"success {rewards} done {dones} accuracy: {(rewards / dones * 100 if dones else 0)}%"
        )

        if dones >= 1:
            break
        # quit()

    # # show gif of images
    print(images[0].shape)
    from moviepy.editor import ImageSequenceClip

    images = [image.astype(np.uint8) for image in images]
    clip = ImageSequenceClip(list(images), fps=20)
    clip.write_gif("test.gif", fps=20)

    # print(policy.action(obs))


def other():
    pass
    # model_output = rt1x_model.apply(
    # variables,
    # processed_obs,
    # act,
    # train=False,
    # rngs={"random": jax.random.PRNGKey(0)},
    # )

    # SEQUENCE_LENGTH = 15
    # NUM_ACTION_TOKENS = 11
    # LAYER_SIZE = 256
    # VOCAB_SIZE = 512
    # NUM_IMAGE_TOKENS = 81
    # # Extract the actions from the model.
    # time_step_tokens = (
    #     NUM_IMAGE_TOKENS + NUM_ACTION_TOKENS
    # )
    # output_logits = jnp.reshape(
    #     model_output, (BATCH_SIZE, SEQUENCE_LENGTH, time_step_tokens, -1)
    # )
    # action_logits = output_logits[:, -1, ...]
    # action_logits = action_logits[:, NUM_IMAGE_TOKENS - 1 : -1]

    # action_logp = jax.nn.softmax(action_logits)
    # action_token = jnp.argmax(action_logp, axis=-1)

    # action_detokenized = detokenize_action(action_token, VOCAB_SIZE, world_vector_range=(-2.0, 2.0))

    # action = np.concatenate([action_detokenized["world_vector"], action_detokenized["rotation_delta"], action_detokenized["gripper_closedness_action"]], axis=1)

    # print(f"Detokenized actions: {action}")

    # images = []
    # while True:
    #   obs =  obs['image'][0]['overhead_camera']['rgb'][None]
    #   processed_obs['image'] = add_seq_dim_padding(obs)
    #   images.append(obs[0])
    #   raw_action, action = policy.action(processed_obs)
    #   del action["terminate_episode"]

    #   action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]], axis=1)

    #   obs, reward, done, _ = env.step(action)
    #   # print("same", np.all(obs['simpler-img'][0] == obs['simpler-img'][1]), obs['simpler-img'][0].shape)
    #   print("raw_action", raw_action)
    #   print("action", action)

    #   # if np.all(prev_obs == obs['simpler-img']):
    #     # print("ENV IS NOT STEPPING")

    #   done_count += int(sum(done))
    #   success_count += int(sum(reward))

    #   print("success", success_count, "\tdone", done_count, "\naccuracy:", success_count / done_count * 100 if done_count > 0 else 0, "%")

    #   if done_count >= 1:
    #     break
    #   # quit()

    # # show gif of images
    # print(images[0].shape)
    # from moviepy.editor import ImageSequenceClip
    # images = [image.astype(np.uint8) for image in images]
    # clip = ImageSequenceClip(list(images), fps=20)
    # clip.write_gif('test.gif', fps=20)

    # print(policy.action(obs))


if __name__ == "__main__":
    main()


def sandbox(cfg):
    pass


#   BATCH_SIZE = 2

#   SEQUENCE_LENGTH = 15
#   NUM_ACTION_TOKENS = 11
#   LAYER_SIZE = 256
#   VOCAB_SIZE = 512
#   NUM_IMAGE_TOKENS = 81

#   rt1x_model = RT1(
#       num_image_tokens=NUM_IMAGE_TOKENS,
#       num_action_tokens=NUM_ACTION_TOKENS,
#       layer_size=LAYER_SIZE,
#       vocab_size=VOCAB_SIZE,
#       # Use token learner to reduce tokens per image to 81.
#       use_token_learner=True,
#       # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
#       world_vector_range=(-2.0, 2.0)
#   )

#   obs = {
#       "image": jnp.ones((BATCH_SIZE, 15, 300, 300, 3)),
#       "natural_language_embedding": jnp.ones((BATCH_SIZE, 15, 512)),
#   }
#   act = {
#       "world_vector": jnp.ones((BATCH_SIZE, 15, 3)),
#       "rotation_delta": jnp.ones((BATCH_SIZE, 15, 3)),
#       "gripper_closedness_action": jnp.ones((BATCH_SIZE, 15, 1)),
#       "base_displacement_vertical_rotation": jnp.ones((BATCH_SIZE, 15, 1)),
#       "base_displacement_vector": jnp.ones((BATCH_SIZE, 15, 2)),
#       "terminate_episode": jnp.ones((BATCH_SIZE, 15, 3), dtype=jnp.int32),
#   }

#   variables = rt1x_model.init(
#       {
#           "params": jax.random.PRNGKey(0),
#           "random": jax.random.PRNGKey(0),
#       },
#       obs,
#       act,
#       train=False,
#   )
#   model_output = rt1x_model.apply(
#       variables,
#       obs,
#       act,
#       train=False,
#       rngs={"random": jax.random.PRNGKey(0)},
#   )

#   # Extract the actions from the model.
#   time_step_tokens = (
#       NUM_IMAGE_TOKENS + NUM_ACTION_TOKENS
#   )
#   output_logits = jnp.reshape(
#       model_output, (BATCH_SIZE, SEQUENCE_LENGTH, time_step_tokens, -1)
#   )
#   action_logits = output_logits[:, -1, ...]
#   action_logits = action_logits[:, NUM_IMAGE_TOKENS - 1 : -1]

#   action_logp = jax.nn.softmax(action_logits)
#   action_token = jnp.argmax(action_logp, axis=-1)

#   action_detokenized = detokenize_action(action_token, VOCAB_SIZE, world_vector_range=(-2.0, 2.0))
#   print(f"Detokenized actions: {action_detokenized}")

# if __name__ == "__main__":
#     main()

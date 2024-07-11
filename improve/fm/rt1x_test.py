import copy
import os.path as osp
from collections import deque
from dataclasses import asdict, dataclass
from pprint import pprint

import jax
import jax.numpy as jnp
import numpy as np
import simpler_env as simpler
import tensorflow as tf
import tensorflow_hub as hub
from flax.training import checkpoints
from stable_baselines3.common.vec_env import SubprocVecEnv
from transforms3d.euler import euler2axangle

import hydra
import improve
import improve.hydra.resolver
from improve.oxe_rt.rt1_model import RT1, detokenize_action
from improve.wrapper import dict_util as du


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

    rt1x_model = RT1(**asdict(RT1ModelConfig()))
    policy = RT1Policy(
        **asdict(RT1PolicyConfig()),
        model=rt1x_model,
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
    env = eval_env

    #
    #
    #

    instructions = env.env_method("get_language_instruction")
    policy.reset(instructions)

    obs = env.reset()

    # dummy = { "image": jnp.ones((BATCH_SIZE, 15, 300, 300, 3)), "natural_language_embedding": jnp.ones((BATCH_SIZE, 15, 512)), }

    rewards, dones = 0, 0
    while True:
        obs = obs["simpler-img"]

        # untranspose = lambda x: np.transpose(x, (0, 2, 3, 1))
        # obs = untranspose(obs)

        # raw_action, action = policy.action(dummy)
        raw_action, action = policy.step(obs)
        action = np.concatenate(
            [action["world_vector"], action["rot_axangle"], action["gripper"]], axis=-1
        )

        obs, reward, done, _ = env.step(action)
        # print("same", np.all(obs['simpler-img'][0] == obs['simpler-img'][1]), obs['simpler-img'][0].shape)
        # print("raw_action", raw_action)
        print("action", action)

        # if np.all(prev_obs == obs['simpler-img']):
        # print("ENV IS NOT STEPPING")

        rewards += float(sum(reward))
        dones += float(sum(done))

        print(
            f"success {rewards} done {dones} accuracy: {(rewards / dones * 100 if dones else 0)}%"
        )

        #  if dones >= 1:
        #  break
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

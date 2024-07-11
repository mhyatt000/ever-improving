import json
import os.path as osp
from dataclasses import asdict

from improve import cn
from improve.fm.batch_octo import BatchedOctoInference
from improve.fm.rtx import RT1Policy


def build_foundation_model(fmcn: cn.FoundationModel):
    """Builds the model."""

    if fmcn.policy in ["rt1", "rtx"]:
        from simpler_env.policies.rt1.rt1_model import RT1Inference

        # model = RT1Inference(saved_model_path=fmcn.ckpt, policy_setup=policy_setup)
        model = RT1Policy(
            model=fmcn.model,
            ckpt=fmcn.ckpt,
            seqlen=fmcn.seqlen,
            batch_size=fmcn.batch_size,
            policy_setup=fmcn.policy_setup,
            cached=fmcn.cached,
            task=fmcn.task,
        )

    elif "octo" in fmcn.policy:
        from simpler_env.policies.octo.octo_model import OctoInference

        model = BatchedOctoInference(
            batch_size=fmcn.batch_size,
            model_type=fmcn.ckpt,
            policy_setup=fmcn.policy_setup,
        )
        # model = OctoInference(model_type=fmcn.ckpt, policy_setup=policy_setup)

    else:
        raise NotImplementedError()

    return model


from omegaconf import OmegaConf as OC

import hydra
import improve
import improve.hydra.resolver
from improve import cn


@hydra.main(config_path=improve.CONFIG, config_name="config", version_base="1.3.2")
def main(cfg):

    # em = load_task_embedding("google_robot_pick_horizontal_coke_can")
    # print(em)
    # quit()

    from improve.env import make_env

    # model = build_foundation_model(fmcn)
    import numpy as np

    import simpler_env as simpler
    env = simpler.make(
        cfg.env.foundation.task,
        # cant find simpler-img if you specify the mode
        render_mode="cameras",
        # max_episode_steps=max_episode_steps,
        renderer_kwargs={
            "offscreen_only": True,
            "device": "cuda:0",
        },
    )


    # env = make_env(cfg)()

    env.reset()
    obs, *things = env.step(env.action_space.sample())
    print(obs.keys())
    # model.reset([env.get_language_instruction()])
    return

    imgs = np.random.rand(batch_size, 224, 224, 3).astype(np.uint8)

    model.reset()
    out = model.step(imgs)

    print(out)
    print("done")


if __name__ == "__main__":
    main()

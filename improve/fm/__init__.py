from improve.config import FoundationModel_CN, OctoS_CN

from improve.fm.batch_octo import BatchedOctoInference


def build_foundation_model(fmcn: FoundationModel_CN):
    """Builds the model."""

    # build policy
    if "google_robot" in fmcn.task:
        policy_setup = "google_robot"
    elif "widowx" in fmcn.task:
        policy_setup = "widowx_bridge"
    else:
        raise NotImplementedError()

    if fmcn.policy == "rt1":
        from simpler_env.policies.rt1.rt1_model import RT1Inference

        model = RT1Inference(saved_model_path=fmcn.ckpt, policy_setup=policy_setup)

    elif "octo" in fmcn.policy:
        from simpler_env.policies.octo.octo_model import OctoInference

        if fmcn.n_envs > 1:
            model = BatchedOctoInference(
                batch_size=fmcn.n_envs,
                model_type=fmcn.ckpt,
                policy_setup=policy_setup,
            )
        else:
            model = OctoInference(
                model_type=fmcn.ckpt, policy_setup=policy_setup
            )

    else:
        raise NotImplementedError()

    return model

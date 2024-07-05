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

        model = BatchedOctoInference(
            batch_size=fmcn.batch_size,
            model_type=fmcn.ckpt,
            policy_setup=policy_setup,
        )
        # model = OctoInference(model_type=fmcn.ckpt, policy_setup=policy_setup)

    else:
        raise NotImplementedError()

    return model


def main():
    batch_size = 1
    model = build_foundation_model(
        OctoS_CN(
            batch_size=batch_size,
        )
    )
    import numpy as np

    imgs = np.random.rand(batch_size, 224, 224, 3).astype(np.uint8)

    model.reset(["put the eggplant in the basket"] * batch_size)
    out = model.step(imgs)

    print(out)
    print("done")


if __name__ == "__main__":
    main()

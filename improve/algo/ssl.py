from .base import Algo

class SelfSupervised(Algo):

    def step(self, batch):

        self.model.train()
        self.optimizer.zero_grad()

        """
        'observation': {'agent_partial-action': (torch.float64,
                                              torch.Size([8, 10, 7])),
                     'agent_qpos': (torch.float32, torch.Size([8, 10, 8])),
                     'agent_qvel': (torch.float32, torch.Size([8, 10, 8])),
                     'simpler-img': (torch.uint8,
                                     torch.Size([8, 10, 480, 640, 3]))},
        'reward': (torch.float64, torch.Size([8, 10])),
        """

        # TODO can this be a transform for the dataset?
        # preprocess before prefetching
        img = self.preprocessor._process(
            batch["observation"]["simpler-img"], static=True, train=True
        )

        # xyq quarternions
        # state = batch["observation"]["agent_qpos"]
        # this is wrong
        # state = {"arm": state[:, :7], "gripper": state[:, 7:]}

        # TODO no wrist images rn
        # batch["rgb_static"], batch["rgb_gripper"] = self.preprocessor.rgb_process( batch["rgb_static"], batch["rgb_gripper"], train=True)

        # obs_mask = batch["mask"][..., 0]
        bs, seq = img.shape[:2]
        attn_mask = torch.ones((bs, seq, 1)).to(self.device)

        text = self.tokenizer("put eggplant in the sink").to(self.device)
        text = text.view(1, -1).expand(bs, -1).to(self.device)

        action = batch["observation"]["agent_partial-action"].float()
        obs = {
            "rgb": img,
            # xyz and quarternions for us... or xyz and rpy
            "state": {"arm": action[:, :, :-1], "gripper": action[:, :, -1:]},
            "language": text,
            "mask": attn_mask,
        }

        predictions, targets = self.model(obs)

        action = torch.roll(action, -1, 1).view(bs, seq, 1, -1).repeat(1, 1, 10, 1)
        targets["arm"] = action[..., :-1]
        targets["gripper"] = (action[..., -1:] / 2) + 0.5

        loss = self.model.loss(
            predictions,
            targets,
            batch,
            skip_frame=self.cfg.model_other.skip_frame,
            arm_loss_ratio=self.cfg.training.arm_loss_ratio,
        )

        self.acc.backward(loss["total"])
        self.optimizer.step()
        self.scheduler.step()
        self.log({"loss": loss}, self.nstep)

        if self.nstep % 500 == 0:
            with torch.no_grad():
                values, _ = self.model.MO.value_net._predict(
                    predictions["value"]["input"]
                )
                plot_values(values, self.nstep)

        lr = self.optimizer.param_groups[0]["lr"]
        self.log({"train/lr": lr}, self.nstep)
        self.nstep += 1

        batch, load_time = self.loader.next()
        return batch, load_time

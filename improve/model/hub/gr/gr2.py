import os
import os.path as osp
from pprint import pprint

import clip
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import (DistributedDataParallelKwargs,
                              InitProcessGroupKwargs)
from einops import rearrange, repeat
from flamingo_pytorch import PerceiverResampler
from omegaconf import OmegaConf as OC
# from tensordict import TensorDict
# from tensordict.nn import TensorDictModule as TDM
from torch.distributions import Normal
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from transformers import GPT2Model, get_cosine_schedule_with_warmup

import improve
import improve.model.vision_transformer as vits
from improve.model.modules.extractor import MultiInExtractor
from improve.model.modules.value_net import QRNet
from improve.model.transformer_utils import get_2d_sincos_pos_embed
from improve.model.vision_transformer import Block
from improve.util.loss import policy_improvement as pimp
from improve.util.loss.mask import old_masked_loss as masked_loss
from improve.util.loss.qr import quantile_huber_loss
from improve.wrapper import dict_util as du


def exists(x):
    return x is not None


class Mask(nn.Module):

    def __init__(
        self,
        hidden_size,
        seq,
        chunk_size,
        n_patch_latents,
        use_hand_rgb,
        act_pred=True,
        fwd_pred=True,
        fwd_pred_hand=True,
    ):
        super(Mask, self).__init__()

        self.hidden_size = hidden_size
        self.seq = seq
        self.chunk_size = chunk_size
        self.n_patch_latents = n_patch_latents
        self.use_hand_rgb = use_hand_rgb

        self.act_pred = act_pred
        self.fwd_pred = fwd_pred
        self.fwd_pred_hand = fwd_pred_hand

        self.lnorm = nn.LayerNorm(self.hidden_size)  # Layer norm for embeddings

    def add_query_attention_masks(self, stack, attn_mask):

        mk_zeros = lambda x: torch.zeros(
            (self.bs, self.seq, x), dtype=torch.long, device=stack.device
        )

        if self.act_pred:
            aq_mask = mk_zeros(self.chunk_size)
        if self.fwd_pred:
            obq_mask = mk_zeros(self.n_patch_latents + 1)
            if self.fwd_pred_hand:
                obhq_mask = mk_zeros(self.n_patch_latents + 1)

        attn_mask = torch.cat(
            [x for x in [attn_mask, aq_mask, obq_mask, obhq_mask] if x is not None],
            dim=2,
        )
        return attn_mask.reshape(self.bs, -1)

    def forward(self, stack, attn_mask):

        self.bs, self.seq, *others = stack.shape
        attn_mask = attn_mask.view(self.bs, self.seq, 1)
        tokens, starts = self.count_tokens()

        attn_mask = attn_mask.repeat(1, 1, tokens["for_mask"])
        attn_mask = self.add_query_attention_masks(stack, attn_mask)

        return stack, attn_mask

    def __call__(self, stack, attn_mask):
        things = self.original(stack, attn_mask)
        return things

    def count_tokens(self):
        """returns the total number of tokens in the model"""

        # Number of tokens
        tokens = {
            "lang": 1,
            "state": 1,
            "patch": self.n_patch_latents,
            "obs": 1,
        }

        total = sum(tokens.values())
        total += tokens["obs"] + tokens["patch"] if self.use_hand_rgb else 0
        tokens["for_mask"] = total

        starts = {}  # start idx of prediction tokens

        if self.act_pred:
            starts["act"] = total
            total += self.chunk_size

        if self.fwd_pred:
            starts["obs"] = total
            total += tokens["patch"] + tokens["obs"]

            if self.fwd_pred_hand:
                starts["obs_hand"] = total
                total += tokens["patch"] + tokens["obs"]

        tokens["total"] = total
        return tokens, starts

    def original(self, stack, _attn_mask):

        tokens, starts = self.count_tokens()

        # Layer norm
        stack = stack.reshape(self.bs, tokens["total"] * self.seq, self.hidden_size)
        stack = self.lnorm(stack)

        # Attention mask
        mask = _attn_mask.view(self.bs, self.seq, 1)
        mask = mask.repeat(1, 1, tokens["for_mask"])

        if self.act_pred:
            act_query_attention_mask = torch.zeros(
                (self.bs, self.seq, self.chunk_size),
                dtype=torch.long,
                device=stack.device,
            )
            mask = torch.cat((mask, act_query_attention_mask), dim=2)

        if self.fwd_pred:
            obs_query_attention_mask = torch.zeros(
                (self.bs, self.seq, tokens["patch"] + tokens["obs"]),
                dtype=torch.long,
                device=stack.device,
            )
            mask = torch.cat((mask, obs_query_attention_mask), dim=2)

            if self.fwd_pred_hand:
                obs_hand_query_attention_mask = torch.zeros(
                    (self.bs, self.seq, tokens["patch"] + tokens["obs"]),
                    dtype=torch.long,
                    device=stack.device,
                )
                mask = torch.cat((mask, obs_hand_query_attention_mask), dim=2)

        mask = mask.reshape(self.bs, tokens["total"] * self.seq)
        return stack, mask, tokens, starts


class ActionHead(nn.Module):

    def __init__(self, hidden_size, act_dim, chunk_size):
        super(ActionHead, self).__init__()

        self.hidden_size = hidden_size
        self.act_dim = act_dim
        self.chunk_size = chunk_size

        hid2 = self.hidden_size // 2
        self.mlps = nn.Sequential(
            nn.Linear(self.hidden_size, hid2), nn.Linear(hid2, hid2)
        )
        self.arm = nn.Linear(hid2, 2 * (self.act_dim - 1))
        self.gripper = nn.Linear(hid2, 2 * 1)

    def normal(self, x):
        x = rearrange(x, "b t n (c d) -> b t n c d", c=2)
        # x = x[..., 0, :, :].unsqueeze(-3)
        return Normal(x[..., 0, :], torch.clamp(x[..., 1, :], min=1e-3))

    def forward(self, x, tokens, starts):
        """predict the actions

        uses action chunking to predict the actions across the sequence
        added sigmoid since gripper is binary in the dataset

        Args:
            x (torch.Tensor): input tensor
            tokens (dict): number of tokens
            starts (dict): start index of prediction tokens
        Returns:
            arm (torch.Tensor (b, t, chunk_size, act_dim-1)): predicted arm actions
            gripper (torch.Tensor): predicted gripper actions
        """

        start, end = starts["act"], starts["act"] + self.chunk_size
        embeds = x[:, :, start:end, :]

        embeds = self.mlps(embeds)

        arm = self.normal(self.arm(embeds))
        gripper = self.normal(F.sigmoid(self.gripper(embeds)))

        out = {
            "arm": arm,
            "gripper": gripper,
        }
        return out


class MultiOutHead(nn.Module):

    def __init__(
        self,
        hidden_size,
        act_dim,
        patch_size,
        image_size,
        chunk_size,
        fwd_pred,
        fwd_pred_hand,
        act_pred,
        n_patch_latents,
    ):
        super(MultiOutHead, self).__init__()

        self.hidden_size = hidden_size
        self.act_dim = act_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.chunk_size = chunk_size

        self.act_pred = act_pred
        self.fwd_pred = fwd_pred
        self.fwd_pred_hand = fwd_pred_hand

        self.n_patch_latents = n_patch_latents

        self.init_forward_prediction()

        # value head
        self.value_net = QRNet(
            num_quantiles=32,
            num_actions=10,
            act_dim=act_dim,
            dim=384,  # 192,
            depth=6,
            dim_head=64,
            heads=8,
            num_latents=64,
            num_media_embeds=4,
        )

        self.head = nn.ModuleDict(
            {
                "action": ActionHead(hidden_size, act_dim, chunk_size),
                "value": self.value_net,
            }
        )

        # we'll need the log-prob for the numerator of the importance weights
        # self.head["policy"] = ProbabilisticActor( module=self.head["action"], in_keys=["loc", "scale"], distribution_class=TanhNormal, distribution_kwargs={"min": -1, "max": 1}, return_log_prob=True,)

    def init_forward_prediction(self):

        self.decoder_embed = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_size))

        block = lambda: Block(
            self.hidden_size,
            16,
            4,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
        )

        decoder_depth = 2
        self.decoder_blocks = nn.ModuleList([block() for i in range(decoder_depth)])

        self.decoder_norm = nn.LayerNorm(self.hidden_size)
        # decoder to patch
        self.decoder_pred = nn.Linear(
            self.hidden_size, self.patch_size**2 * 3, bias=True
        )

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, (self.image_size // self.patch_size) ** 2, self.hidden_size),
            requires_grad=False,
        )  # (1, n_patch, h)

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], (self.image_size // self.patch_size)
        )

        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

    def predict_actions(self, x, tokens, starts):
        """predict the actions

        uses action chunking to predict the actions across the sequence
        added sigmoid since gripper is binary in the dataset

        Args:
            x (torch.Tensor): input tensor
            tokens (dict): number of tokens
            starts (dict): start index of prediction tokens
        Returns:
            arm (torch.Tensor (b, t, chunk_size, act_dim-1)): predicted arm actions
            gripper (torch.Tensor (b, t, chink_size, 1)): predicted gripper actions
        """

        return self.head["action"](x, tokens, starts)

        arm, gripper = None, None
        if self.act_pred:
            start, end = starts["act"], starts["act"] + self.chunk_size
            action_embedding = x[:, :, start:end, :]

            for pred_act_mlp in self.pred_act_mlps:
                action_embedding = pred_act_mlp(action_embedding)

            arm_action_preds = self.pred_arm_act(action_embedding)
            gripper_action_preds = F.sigmoid(self.pred_gripper_act(action_embedding))

        return arm, gripper

    def predict_forward(self, x):

        # print("decoder")
        # print(x.shape)

        obs_preds, obs_hand_preds = None, None

        if self.fwd_pred:
            mask_tokens = self.mask_token.repeat(
                self.bs,
                self.seq,
                (self.image_size // self.patch_size) ** 2,
                1,
            )
            pos = self.decoder_pos_embed.unsqueeze(0).repeat(self.bs, self.seq, 1, 1)
            mask_tokens = mask_tokens + pos

            # print(mask_tokens.shape)
            obs_preds = self.decode_predictions(x, mask_tokens, is_hand=False)
            # print(obs_preds.shape)

        if self.fwd_pred_hand:
            obs_hand_preds = self.decode_predictions(x, mask_tokens, is_hand=True)

        return obs_preds, obs_hand_preds

    def decode_predictions(self, x, mask_tokens, is_hand=False):

        start_idx = -self.n_patch_latents - 1 if is_hand else -self.chunk_size

        obs_pred = self.decoder_embed(x[:, :, start_idx:])
        obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=2).reshape(
            -1, obs_pred.shape[-2], obs_pred.shape[-1]
        )

        for blk in self.decoder_blocks:
            obs_pred_ = blk(obs_pred_)
            # print("blk", obs_pred_.shape)
        obs_pred_ = self.decoder_norm(obs_pred_)
        obs_preds = self.decoder_pred(obs_pred_).reshape(
            self.bs, self.seq, -1, obs_pred_.shape[-1]
        )

        return obs_preds[:, :, -self.n_patch_latents :]

    def old_forward(self, x, tokens, starts):

        obs, obs_hand = self.predict_forward(x)

        predictions = {
            "arm": arm,
            "gripper": gripper,
            "obs": obs,
            "obs_hand": obs_hand,
            "values": values,
        }
        return predictions

    def forward(self, x, tokens, starts, batch_act):
        """NOTE: only attend to the correct tokens with no cheating"""

        # Action prediction
        arm, gripper = self.predict_actions(x, tokens, starts).values()

        # Forward prediction
        if self.fwd_pred:
            mask_token = self.mask_token  # (1, 1, 1, h)
            mask_tokens = mask_token.repeat(
                self.bs,
                self.seq,
                (self.image_size // self.patch_size) ** 2,
                1,
            )  # (b, l, n_patches, h)
            mask_tokens = mask_tokens + self.decoder_pos_embed.unsqueeze(0).repeat(
                self.bs, self.seq, 1, 1
            )  # (b, l, n_patches, h)

            start = starts["obs"]
            end = starts["obs"] + self.n_patch_latents + tokens["obs"]
            # (b, l, n_patch_latents + 1, h)
            obs_pred = self.decoder_embed(x[:, :, start:end, :])

            # (b, l, n_patches + n_patch_latens + 1, h)
            obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=2)
            # (b * l, n_patches + n_patch_latens + 1, h)
            obs_pred_ = obs_pred_.reshape(-1, obs_pred_.shape[-2], obs_pred_.shape[-1])

            for blk in self.decoder_blocks:
                obs_pred_ = blk(obs_pred_)

            obs_pred_ = self.decoder_norm(obs_pred_)
            # (b * l, n_patches + n_patch_latens + 1, h)
            obs_preds = self.decoder_pred(obs_pred_)
            # (b, l, n_patches + n_patch_latens + 1, h)
            obs_preds = obs_preds.reshape(self.bs, self.seq, -1, obs_preds.shape[-1])
            # (b, l, n_patches, h)
            obs_preds = obs_preds[:, :, (self.n_patch_latents + tokens["obs"]) :]

            obs_hand_preds = None
            if self.fwd_pred_hand:
                start = starts["obs_hand"]
                end = starts["obs_hand"] + self.n_patch_latents + tokens["obs"]
                obs_pred_hand = self.decoder_embed(x[:, :, start:end, :])
                obs_pred_hand_ = torch.cat([obs_pred_hand, mask_tokens], dim=2)
                obs_pred_hand_ = obs_pred_hand_.reshape(
                    -1, obs_pred_hand_.shape[-2], obs_pred_hand_.shape[-1]
                )

                for blk in self.decoder_blocks:
                    obs_pred_hand_ = blk(obs_pred_hand_)
                obs_pred_hand_ = self.decoder_norm(obs_pred_hand_)
                obs_hand_preds = self.decoder_pred(obs_pred_hand_)
                obs_hand_preds = obs_hand_preds.reshape(
                    self.bs, self.seq, -1, obs_hand_preds.shape[-1]
                )
                obs_hand_preds = obs_hand_preds[
                    :, :, (self.n_patch_latents + tokens["obs"]) :
                ]

        # Value prediction
        start = starts["obs"]
        end = starts["obs"] + self.n_patch_latents + tokens["obs"]
        # (b, l, n_patch_latents + 1, h)
        obs_value_tokens = x[:, :, start:end, :]
        value = self.value_net(obs_value_tokens, batch_act)

        # sampleten = lambda x: torch.cat([x.rsample() for _ in range(10)], dim=-2)
        # actions = torch.cat([sampleten(arm), sampleten(gripper)], dim=-1)
        actions = torch.cat([arm.rsample(), gripper.rsample()], dim=-1)
        value_improve = self.value_net(obs_value_tokens, actions)

        log_probs = {
            "action": arm.log_prob(actions[..., :-1]),
            "gripper": gripper.log_prob(actions[..., -1].unsqueeze(-1)),
        }
        log_probs["all"] = torch.cat(
            [log_probs["action"], log_probs["gripper"]], dim=-1
        )

        predictions = {
            "arm": arm,
            "gripper": gripper,
            "log_prob": log_probs,
            "obs": obs_preds,
            "obs_hand": obs_hand_preds,
            "value": {
                "input": value,
                "improve": value_improve,
            },
        }
        return predictions


class GR2(nn.Module):

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        seq_len,
        chunk_size,
        training_target,
        img_feat_dim,
        patch_feat_dim,
        lang_feat_dim,
        resampler_params,
        without_norm_pixel_loss=False,
        use_hand_rgb=True,
        pretrained=None,
        **kwargs
    ):
        super(GR2, self).__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.seq = seq_len
        self.chunk_size = chunk_size

        # GPT
        self.hidden_size = hidden_size

        self.n_patches = 49
        self.patch_size = 16
        self.image_size = 224  # TODO: make this a parameter

        self.img_feat_dim = img_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.patch_feat_dim = patch_feat_dim
        self.use_hand_rgb = use_hand_rgb

        self.act_pred = training_target["act_pred"]
        self.fwd_pred = training_target["fwd_pred"]
        self.fwd_pred_hand = training_target["fwd_pred_hand"]

        self.without_norm_pixel_loss = without_norm_pixel_loss or False

        self.MI = MultiInExtractor(
            mae=pretrained["visual"],
            llm=pretrained["language"],
            patch_feat_dim=patch_feat_dim,
            resampler_params=resampler_params,
            state_dim=state_dim,
            hidden_size=hidden_size,
            lang_feat_dim=lang_feat_dim,
            img_feat_dim=img_feat_dim,
            patch_size=self.patch_size,
            without_norm_pixel_loss=self.without_norm_pixel_loss,
        )
        self.time_emb = nn.Embedding(self.seq, self.hidden_size)

        self.n_patch_latents = resampler_params["num_latents"]

        self.query = nn.ModuleDict(
            {
                "action": nn.Embedding(1, self.hidden_size),
                "action_chunk": nn.Embedding(self.chunk_size, self.hidden_size),
                "obs": nn.Embedding(self.n_patch_latents + 1, self.hidden_size),
                "obs_hand": nn.Embedding(self.n_patch_latents + 1, self.hidden_size),
            }
        )
        self.query["action_chunk"].weight.data.fill_(0)  # finetune it from zero weight

        self.use_query = {
            "action": False,
            "obs": False,
            "obs_hand": False,
        }

        self.mask = Mask(
            hidden_size=self.hidden_size,
            seq=self.seq,
            chunk_size=self.chunk_size,
            n_patch_latents=self.n_patch_latents,
            use_hand_rgb=self.use_hand_rgb,
            #
            act_pred=self.act_pred,
            fwd_pred=self.fwd_pred,
            fwd_pred_hand=self.fwd_pred_hand,
        )

        self.lnorm = nn.LayerNorm(self.hidden_size)  # Layer norm for embeddings

        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
        self.transformer = GPT2Model(config)

        self.MO = MultiOutHead(
            hidden_size=hidden_size,
            act_dim=act_dim,
            patch_size=self.patch_size,
            image_size=self.image_size,
            chunk_size=self.chunk_size,
            fwd_pred=self.fwd_pred,
            fwd_pred_hand=self.fwd_pred_hand,
            act_pred=self.act_pred,
            n_patch_latents=self.n_patch_latents,
        )

    @classmethod
    def from_hydra(cls, cn, pretrained=None):

        resampler_params = {
            "depth": cn.perceiver.resampler_depth,
            "dim_head": cn.perceiver.resampler_dim_head,
            "heads": cn.perceiver.resampler_heads,
            "num_latents": cn.perceiver.resampler_num_latents,
            "num_media_embeds": cn.perceiver.resampler_num_media_embeds,
        }

        model = cls(
            state_dim=cn.state_dim,
            act_dim=cn.act_dim,
            hidden_size=cn.embed_dim,
            seq_len=cn.seq_len,
            chunk_size=cn.chunk_size,
            training_target=cn.training_target,
            img_feat_dim=cn.img_feat_dim,
            patch_feat_dim=cn.patch_feat_dim,
            lang_feat_dim=cn.lang_feat_dim,
            resampler_params=resampler_params,
            without_norm_pixel_loss=cn.without_norm_pixel_loss,
            use_hand_rgb=cn.use_hand_rgb,
            **OC.to_container(cn.gpt_kwargs, resolve=True),
            pretrained=pretrained,
        )
        model.cn = cn
        return model

    def safe_cat(tensor_dict):
        stacked_dict = {}
        for k, v in tensor_dict.items():
            if v is not None:
                stacked_dict[k] = torch.stack(v)
        return stacked_dict

    def stack_embeddings(self, embeddings):
        # Format sequence: lang, state, patch, obs, hand_patch, hand_obs, [ACT], [OBS], [OBS_HAND]

        reshape = lambda x: (
            x.view(self.bs, self.seq, -1, self.hidden_size) if x is not None else None
        )
        embeddings = du.apply(embeddings, reshape)

        stack = [v for v in embeddings.values() if v is not None]
        stack = torch.cat(stack, dim=2)

        return self.maybe_add_queries(stack)

    def maybe_add_queries(self, stack):

        expand = lambda x: x.repeat(self.bs, self.seq, 1, 1)
        reshape_query = lambda x, sza, szb: expand(x.view(1, 1, sza, szb))

        acq, obq, obhq = None, None, None
        if self.act_pred:
            aq = self.query["action"].weight
            acq = self.query["action_chunk"].weight + aq
            acq = reshape_query(acq, self.chunk_size, self.hidden_size)

        if self.fwd_pred:
            obq = self.query["obs"].weight
            obq = reshape_query(obq, self.n_patch_latents + 1, self.hidden_size)

            if self.fwd_pred_hand:
                obhq = self.query["obs_hand"].weight
                obhq = reshape_query(obhq, self.n_patch_latents + 1, self.hidden_size)

        stack = torch.cat([x for x in [stack, acq, obq, obhq] if x is not None], dim=2)
        return stack

    def add_timestep_emb(self, embed):
        """adds time dimension to the language and patch embeddings"""

        time = self.time_emb.weight
        embed["language"] = embed["language"].view(self.bs, 1, -1)

        embed["rgb_patch"] = embed["rgb_patch"] + time.view(
            self.seq, 1, self.hidden_size
        )

        if "hand_rgb_patch" in embed:
            raise NotImplementedError

        _embed = {k: v for k, v in embed.items() if "patch" not in k}
        _embed = du.apply(_embed, lambda x: x + self.time_emb.weight)
        embed.update(_embed)

        return embed

    def _stack(self, embed, attn_mask):

        # attn_mask is [19,10,1] here
        # should it be [19,10,1,1]?

        embed = self.add_timestep_emb(embed)
        stack = self.stack_embeddings(embed)

        stack, attn_mask, tokens, starts = self.mask(stack, attn_mask)
        return stack, attn_mask, tokens, starts

    def transformer_forward(self, stack, attn_mask):

        stack = stack.reshape(self.bs, -1, self.hidden_size)
        stack = self.lnorm(stack)
        out = self.transformer(inputs_embeds=stack, attention_mask=attn_mask)
        return out["last_hidden_state"].reshape(self.bs, self.seq, -1, self.hidden_size)

    def forward(self, batch):

        self.use_hand_rgb = False  # for now
        self.fwd_pred_hand = False

        bs, seq, c, h, w = batch["rgb"].shape
        self.bs, self.seq = bs, seq

        self.mask.bs = bs  # hack
        self.MO.bs, self.MO.seq = bs, seq

        embeddings, targets = self.MI({k: v for k, v in batch.items() if k != "mask"})
        targets["mask"] = batch["mask"]

        stack, attn_mask, tokens, starts = self._stack(embeddings, batch["mask"])

        x = self.transformer_forward(stack, attn_mask)
        # value net has to train on ground truth actions not its own actions
        actions = torch.cat([batch["state"]["arm"], batch["state"]["gripper"]], dim=-1)
        predictions = self.MO(x, tokens, starts, actions)

        return predictions, targets

    def value_loss(self, pred, tgt, batch):

        pred["value"]["input"]
        bs, seq, nact, nquant = pred["value"]["input"].shape
        tgt["value"] = torch.roll(pred["value"]["input"], -1, 1).clone().detach()

        tgt["value"] = batch["reward"].unsqueeze(-1).repeat(1, 1, nact).view(-1, 1) + (
            1 - batch["terminated"].long().unsqueeze(-1).repeat(1, 1, nact).view(-1, 1)
        ) * 0.99 * tgt["value"].view(-1, nquant)

        # only successfull observations contriibute to the loss
        # obs_mask = tgt["mask"][..., 0].repeat(1, 1, nact).view(-1, nact, 1)
        obs_mask = batch["info"]["will_succeed"].repeat(1, 1, nact).view(-1, nact, 1)
        value_loss = masked_loss(
            pred["value"]["input"].view(
                -1, 1, nquant
            ),  # nact is 1 for ground truth actions
            tgt["value"].view(-1, 1, nquant),
            obs_mask,
            0,
            quantile_huber_loss,
        )
        return value_loss

        """
        this vvv was predicting the value of model actions as if it was ground truth
        but it should be predicting the value of the ground truth actions
        might be useful later so dont delete yet
        """

        # these are the values of all the actions
        quantiles = pred["value"]["input"]
        bs, seq, nact, nquant = quantiles.shape

        # these are the values of the optimal actions
        qs, _ = self.MO.value_net._predict(quantiles)
        # predict quantiles for all actions
        # even suboptimal ones
        qs = qs.unsqueeze(-2).expand(-1, -1, nact, -1)

        # TD1 gold standard
        tgt["value"] = torch.roll(qs, -1, 1).clone().detach()

        # print(batch["reward"].view(-1, 1).expand(bs * seq * nact, 1).shape)
        # print(batch["terminated"].long().view(-1, 1).expand(bs * seq * nact, 1).shape)
        # print(tgt["value"].view(-1, nquant).shape)

        # 1-step TD target
        # gets funky for 4D tensors
        tgt["value"] = batch["reward"].unsqueeze(-1).repeat(1, 1, nact).view(-1, 1) + (
            1 - batch["terminated"].long().unsqueeze(-1).repeat(1, 1, nact).view(-1, 1)
        ) * 0.99 * tgt["value"].view(-1, nquant)

        # current_quantiles, _ = model._predict(current_obs, False)
        # # grab the first n_quantiles (200)
        # current_quantiles = current_quantiles.squeeze(dim=2)

        # dont mask if successful
        # obs_mask = batch["info"]["will_succeed"].repeat(1, 1, nact).view(-1, nact, 1)
        obs_mask = tgt["mask"][..., 0]
        value_loss = masked_loss(
            pred["value"].view(-1, nact, nquant),
            tgt["value"].view(-1, nact, nquant),
            obs_mask,
            0,
            quantile_huber_loss,
        )

        return value_loss

    def policy_improvement_loss(self, pred, tgt, batch):
        """incentivize the model to take actions that lead to success"""

        # where eta is a hyperparameter determining the
        # strength of the regularization towards the reference policy
        eta = 0.5  # temperature parameter
        w = pimp.compute_advantage(pred, eta)
        loss = pimp.compute_policy_improvement(pred["log_prob"]["all"], w)

        return loss

    def loss(self, pred, tgt, batch, skip_frame=3, arm_loss_ratio=100):

        obs_mask = tgt["mask"][..., 0]

        loss = {}

        loss["pimp"] = self.policy_improvement_loss(pred, tgt, batch)

        # pprint(du.apply(pred, lambda x: x.shape if x is not None else None))
        # pprint(du.apply(tgt, lambda x: x.shape if x is not None else None))
        # pprint(du.apply(batch, lambda x: x.shape if x is not None else None))

        loss["rgb_static"], loss["rgb_gripper"] = 0, 0
        _masked_loss = lambda x, y: masked_loss(x, y, obs_mask, skip_frame, F.mse_loss)
        # TODO clean naming conventions
        loss["rgb_static"] = _masked_loss(pred["obs"], tgt["rgb"])
        if self.use_hand_rgb and pred["obs_hand"] is not None:
            loss["rgb_gripper"] = _masked_loss(pred["obs_hand"], tgt["obs_hand"])

        """
        chop = lambda x: x[:, 1:-1, :]
        tgt["mask"] = chop(tgt["mask"])
        tgt["arm"] = chop(tgt["arm"])
        tgt["gripper"] = chop(tgt["gripper"])
        pred["arm"] = chop(pred["arm"])
        pred["gripper"] = chop(pred["gripper"])

        print(tgt["arm"].shape, pred["arm"].shape, tgt["mask"].shape)
        """

        obs_mask = batch["info"]["will_succeed"]

        # predictions are distributions now
        # BC loss over the mean
        _masked_loss = lambda x, y: masked_loss(x, y, obs_mask, 0, F.smooth_l1_loss)
        loss["action_arm"] = _masked_loss(pred["arm"].mean, tgt["arm"][..., :6])
        loss["action_gripper"] = _masked_loss(
            pred["gripper"].mean, tgt["gripper"][..., -1:]
        )

        loss["value"] = self.value_loss(pred, tgt, batch)

        loss = du.apply(
            loss,
            lambda x: (0 if torch.isnan(x) else x) if type(x) == torch.Tensor else x,
        )

        alpha = 0.9  # policy improvement loss term
        arm_loss_ratio=100
        loss["total"] = (
            loss["rgb_static"]
            + loss["rgb_gripper"]
            + arm_loss_ratio
            * (
                alpha * loss["action_arm"]
                + alpha * loss["action_gripper"]
                + (1 - alpha) * loss["pimp"]
            )
            + loss["value"]
        )
        return loss

    def step(self, batch):
        pred, tgt = self(batch)
        loss = self.loss(pred, tgt, batch)
        return loss
        pass


@hydra.main(config_path=improve.CONFIG, config_name="gr1_config", version_base="1.3.2")
def main(cfg):

    device = "cuda"  # "cpu"

    model_clip, _ = clip.load(cfg.submodel.clip_backbone, device=device)
    model_mae = vits.__dict__["vit_base"](patch_size=16, num_classes=0).to(device)
    checkpoint = torch.load(osp.join(improve.WEIGHTS, cfg.paths.mae_ckpt))
    model_mae.load_state_dict(checkpoint["model"], strict=False)

    pretrained = {"visual": model_mae, "language": model_clip}
    model = GR2.from_hydra(cfg.model, pretrained).to(device)

    bs, seq = 4, 10
    batch = {
        "rgb": torch.randn(4, 10, 3, 224, 224),
        "mask": torch.randn(4, 10, 1),
        "state": {
            "arm": torch.randn(4, 10, 6),
            "gripper": torch.randn(4, 10, 1),
        },
        # "obs_hand": torch.randn(4, 10, 196, 768),
        # "obs": torch.randn(4, 10, 196, 768),
        # "info": {"will_succeed": torch.randn(4, 10, 1)},
        "language": torch.ones([4, 77], dtype=torch.long),
    }

    batch = du.apply(batch, lambda x: x.to(device) if exists(x) else None)

    pred, targets = model(batch)

    batch["info"] = {"will_succeed": torch.randn(4, 10, 1)}
    batch["reward"] = torch.randn(4, 10)
    batch["terminated"] = torch.randn(4, 10)

    action = torch.randn(4, 10, 7)
    action = torch.roll(action, -1, 1).view(bs, seq, 1, -1).repeat(1, 1, 10, 1)
    targets["arm"] = action[..., :-1]
    targets["gripper"] = (action[..., -1:] / 2) + 0.5

    batch = du.apply(batch, lambda x: x.to(device) if exists(x) else None)
    targets = du.apply(targets, lambda x: x.to(device) if exists(x) else None)

    loss = model.loss(pred, targets, batch)
    pprint(loss)
    # pprint(du.apply(pred, lambda x: x.shape if exists(x) else None))


if __name__ == "__main__":
    main()

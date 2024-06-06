import os
import os.path as osp
from pprint import pprint

import clip
import hydra
import improve
import improve.pac.gr1.models.vision_transformer as vits
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import (DistributedDataParallelKwargs,
                              InitProcessGroupKwargs)
from flamingo_pytorch import PerceiverResampler
from improve.pac.gr1.models.modules.extractor import MultiInExtractor
from improve.pac.gr1.models.transformer_utils import get_2d_sincos_pos_embed
from improve.pac.gr1.models.vision_transformer import Block
from improve.pac.gr1.util.loss import masked_loss
from improve.wrapper import dict_util as du
from omegaconf import OmegaConf as OC
from transformers import GPT2Model, get_cosine_schedule_with_warmup


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

        self.init_action_prediction()
        self.init_forward_prediction()

        self.head = {
            "arm": None,  # yet
            "gripper": None,  # yet
        }

    def init_action_prediction(self):

        hid2 = self.hidden_size // 2
        self.pred_act_mlps = nn.ModuleList(
            [nn.Linear(self.hidden_size, hid2), nn.Linear(hid2, hid2)]
        )
        self.pred_arm_act = nn.Linear(hid2, self.act_dim - 1)  # arm action
        self.pred_gripper_act = nn.Linear(hid2, 1)  # gripper action (binary)

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

    def predict_actions(self, x):

        arm, gripper = None, None

        if self.act_pred:

            latent = x[:, :, -self.chunk_size :]
            for pred_act_mlp in self.pred_act_mlps:
                latent = pred_act_mlp(latent)

            arm = self.pred_arm_act(latent)
            gripper = self.pred_gripper_act(latent)

        return arm, gripper

    def predict_forward(self, x):

        print("decoder")
        print(x.shape)

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

            print(mask_tokens.shape)
            obs_preds = self.decode_predictions(x, mask_tokens, is_hand=False)
            print(obs_preds.shape)

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
            print("blk", obs_pred_.shape)
        obs_pred_ = self.decoder_norm(obs_pred_)
        obs_preds = self.decoder_pred(obs_pred_).reshape(
            self.bs, self.seq, -1, obs_pred_.shape[-1]
        )

        return obs_preds[:, :, -self.n_patch_latents :]

    def forward(self, x, tokens, starts):

        arm, gripper = self.predict_actions(x)
        obs, obs_hand = self.predict_forward(x)

        print(obs.shape)

        predictions = {
            "arm": arm,
            "gripper": gripper,
            "obs": obs,
            "obs_hand": obs_hand,
        }
        return predictions

    def original(self, x, tokens, starts):

        print("original")
        print(x.shape)

        # Action prediction
        if self.act_pred:
            action_embedding = x[
                :,
                :,
                starts["act"] : (starts["act"] + self.chunk_size),
            ]

            for pred_act_mlp in self.pred_act_mlps:
                action_embedding = pred_act_mlp(action_embedding)
            arm_action_preds = self.pred_arm_act(
                action_embedding
            )  # (b, t, chunk_size, act_dim - 1)
            gripper_action_preds = self.pred_gripper_act(
                action_embedding
            )  # (b, t, chunk_size, 1)

        # Forward prediction
        if self.fwd_pred:
            mask_token = self.mask_token  # (1, 1, 1, h)
            mask_tokens = mask_token.repeat(
                batch_size,
                sequence_length,
                (self.image_size // self.patch_size) ** 2,
                1,
            )  # (b, l, n_patches, h)
            mask_tokens = mask_tokens + self.decoder_pos_embed.unsqueeze(0).repeat(
                batch_size, sequence_length, 1, 1
            )  # (b, l, n_patches, h)

            obs_pred = self.decoder_embed(
                x[
                    :,
                    :,
                    obs_query_token_start_i : (
                        obs_query_token_start_i + self.n_patch_latents + n_obs_tokens
                    ),
                ]
            )  # (b, l, n_patch_latents + 1, h)
            obs_pred_ = torch.cat(
                [obs_pred, mask_tokens], dim=2
            )  # (b, l, n_patches + n_patch_latens + 1, h)
            obs_pred_ = obs_pred_.reshape(
                -1, obs_pred_.shape[-2], obs_pred_.shape[-1]
            )  # (b * l, n_patches + n_patch_latens + 1, h)
            for blk in self.decoder_blocks:
                obs_pred_ = blk(obs_pred_)
            obs_pred_ = self.decoder_norm(obs_pred_)
            obs_preds = self.decoder_pred(
                obs_pred_
            )  # (b * l, n_patches + n_patch_latens + 1, h)
            obs_preds = obs_preds.reshape(
                batch_size, sequence_length, -1, obs_preds.shape[-1]
            )  # (b, l, n_patches + n_patch_latens + 1, h)
            obs_preds = obs_preds[
                :, :, (self.n_patch_latents + n_obs_tokens) :
            ]  # (b, l, n_patches, h)

            if self.fwd_pred_hand:
                obs_pred_hand = self.decoder_embed(
                    x[
                        :,
                        :,
                        obs_hand_query_token_start_i : (
                            obs_hand_query_token_start_i
                            + self.n_patch_latents
                            + n_obs_tokens
                        ),
                    ]
                )
                obs_pred_hand_ = torch.cat([obs_pred_hand, mask_tokens], dim=2)
                obs_pred_hand_ = obs_pred_hand_.reshape(
                    -1, obs_pred_hand_.shape[-2], obs_pred_hand_.shape[-1]
                )
                for blk in self.decoder_blocks:
                    obs_pred_hand_ = blk(obs_pred_hand_)
                obs_pred_hand_ = self.decoder_norm(obs_pred_hand_)
                obs_hand_preds = self.decoder_pred(obs_pred_hand_)
                obs_hand_preds = obs_hand_preds.reshape(
                    batch_size, sequence_length, -1, obs_hand_preds.shape[-1]
                )
                obs_hand_preds = obs_hand_preds[
                    :, :, (self.n_patch_latents + n_obs_tokens) :
                ]

        print("original")
        print(obs_preds.shape)
        quit()


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
        predictions = self.MO(x, tokens, starts)

        return predictions, targets

    def loss(self, pred, tgt, skip_frame=3, arm_loss_ratio=100):
        """
        {'arm_action_preds': torch.Size([4, 10, 10, 6]),
         'gripper_action_preds': torch.Size([4, 10, 10, 1]),
         'obs_hand_preds': torch.Size([4, 10, 196, 768]),
         'obs_hand_targets': torch.Size([4, 10, 196, 768]),
         'obs_preds': torch.Size([4, 10, 196, 768]),
         'obs_targets': torch.Size([4, 10, 196, 768])}
        """

        obs_mask = tgt["mask"][..., 0]

        loss = {}

        pprint(du.apply(pred, lambda x: x.shape if x is not None else None))
        pprint(du.apply(tgt, lambda x: x.shape if x is not None else None))

        loss["rgb_static"], loss["rgb_gripper"] = 0, 0
        # _masked_loss = lambda x, y: masked_loss(x, y, obs_mask, skip_frame, F.mse_loss)
        # loss["rgb_static"] = _masked_loss(pred["obs"], tgt["obs"])
        # loss["rgb_gripper"] = _masked_loss(pred["obs_hand"], tgt["obs_hand"])

        _masked_loss = lambda x, y: masked_loss(x, y, tgt["mask"], 0, F.smooth_l1_loss)
        loss["action_arm"] = _masked_loss(pred["arm"], tgt["arm"][..., :6])
        loss["action_gripper"] = _masked_loss(pred["gripper"], tgt["gripper"][..., -1:])

        loss["total"] = (
            loss["rgb_static"]
            + loss["rgb_gripper"]
            + arm_loss_ratio * loss["action_arm"]
            + loss["action_gripper"]
        )
        return loss


CONFIG = osp.dirname(osp.dirname(__file__))

print(CONFIG)


@hydra.main(config_path=CONFIG, config_name="gr1_config", version_base="1.3.2")
def main(cfg):

    device = "cpu"

    model_clip, _ = clip.load(cfg.submodel.clip_backbone, device=device)
    model_mae = vits.__dict__["vit_base"](patch_size=16, num_classes=0).to(device)
    checkpoint = torch.load(osp.join(improve.WEIGHTS, cfg.paths.mae_ckpt))
    model_mae.load_state_dict(checkpoint["model"], strict=False)

    pretrained = {"visual": model_mae, "language": model_clip}
    model = GR2.from_hydra(cfg.model, pretrained)


if __name__ == "__main__":
    main()

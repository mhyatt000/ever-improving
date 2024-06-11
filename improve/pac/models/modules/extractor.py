import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from clip import clip
from flamingo_pytorch import PerceiverResampler
from improve.wrapper import dict_util as du
from omegaconf import OmegaConf as OC
from transformers import GPT2Model


class LanguageExtractor(nn.Module):

    def __init__(self, llm, lang_feat_dim, hidden_size):
        super(LanguageExtractor, self).__init__()

        self.hidden_size = hidden_size
        self.lang_feat_dim = lang_feat_dim

        self.llm = llm
        self.tokenizer = clip.tokenize
        self.embed = nn.Linear(self.lang_feat_dim, self.hidden_size)

        """Initialize CLIP model and freeze its parameters."""
        if self.llm is not None:
            for _, param in self.llm.named_parameters():
                param.requires_grad = False

    def tokenize(self, str):
        return self.tokenizer(str)

    def forward(self, language):
        lang = self.llm.encode_text(language)
        lang = lang / (lang.norm(dim=1, keepdim=True) + 1e-6)
        return self.embed(lang.float())


class StateExtractor(nn.Module):

    def __init__(self, state_dim, hidden_size):
        super(StateExtractor, self).__init__()

        self.state_dim = state_dim
        self.hidden_size = hidden_size

        self.fc_arm = nn.Linear(self.state_dim, self.hidden_size)

        self.fc_gripper = nn.Linear(1, self.hidden_size)
        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.extractor = {
            "arm": self.fc_arm,
            "gripper": self.fc_gripper,
        }

    def forward(self, state):

        self.bs, self.seq, *other = state["arm"].shape
        view = lambda x: x.view(self.bs, self.seq, -1)
        embeddings = {
            k: view(self.extractor[k](v)) for k, v in state.items() if v is not None
        }

        emb = torch.cat(list(embeddings.values()), dim=2)
        return self.fc(emb)


class VisualExtractor(nn.Module):

    def __init__(
        self,
        mae,
        patch_feat_dim,
        resampler_params,
        patch_size,
        without_norm_pixel_loss,
    ):
        super(VisualExtractor, self).__init__()
        self.mae = mae

        """Initialize MAE model and freeze its parameters."""
        for _, param in self.mae.named_parameters():
            param.requires_grad = False

        self.n_patch_latents = resampler_params["num_latents"]
        self.resampler_params = resampler_params
        self.patch_feat_dim = patch_feat_dim
        self.patch_size = patch_size
        self.without_norm_pixel_loss = without_norm_pixel_loss

        self.perceiver = PerceiverResampler(
            dim=patch_feat_dim,
            depth=resampler_params["depth"],
            dim_head=resampler_params["dim_head"],
            heads=resampler_params["heads"],
            num_latents=self.n_patch_latents,
            num_media_embeds=resampler_params["num_media_embeds"],
        )

    def forward(self, x):

        bs, seq, c, h, w = x.shape
        self.bs, self.seq = bs, seq
        self.h, self.w = h, w

        obs, patch = self.mae(x.view(bs * seq, c, h, w))
        obs = obs.view(bs, seq, -1)

        return obs, patch

    def targets(self, x):
        """Prepares the forward prediction for the given RGB and hand RGB images."""

        p = self.patch_size
        hp, wp = self.h // p, self.w // p

        x = x.reshape(shape=(self.bs, self.seq, 3, hp, p, wp, p))
        target = self.normalize_targets(x.permute(0, 1, 3, 5, 4, 6, 2))
        target = target.reshape(shape=(self.bs, self.seq, hp * wp, (p**2) * 3))

        return target

    def normalize_targets(self, x):
        """Normalizes the target images."""

        if self.without_norm_pixel_loss:
            return x

        x = (x - x.mean(dim=-1, keepdim=True)) / (
            x.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6
        )
        return x

    def process_patch(self, patch):
        patch = self.perceiver(patch.unsqueeze(1)).squeeze(1)
        return patch.view(self.bs, self.seq, self.n_patch_latents, self.patch_feat_dim)


class MultiInExtractor(nn.Module):

    def __init__(
        self,
        mae,
        llm,
        patch_feat_dim,
        resampler_params,
        state_dim,
        hidden_size,
        lang_feat_dim,
        img_feat_dim,
        patch_size,
        without_norm_pixel_loss,
    ):
        super(MultiInExtractor, self).__init__()

        self.img_feat_dim = img_feat_dim
        self.patch_feat_dim = patch_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.hidden_size = hidden_size

        self.resampler_params = resampler_params
        self.llm = llm
        self.mae = mae

        self.visual = VisualExtractor(
            mae,
            patch_feat_dim,
            resampler_params,
            patch_size=patch_size,
            without_norm_pixel_loss=without_norm_pixel_loss,
        )

        self.extractor = nn.ModuleDict(
            {
                "rgb": self.visual,
                "hand_rgb": self.visual,
                "state": StateExtractor(
                    state_dim=state_dim,
                    hidden_size=hidden_size,
                ),
                "language": LanguageExtractor(
                    llm=llm,
                    lang_feat_dim=lang_feat_dim,
                    hidden_size=hidden_size,
                ),
            }
        )

        self.visual_keys = [
            k for k in self.extractor if type(self.extractor[k]) == VisualExtractor
        ]

        self.embed_fc = nn.ModuleDict(
            {
                "rgb": nn.Linear(self.img_feat_dim, self.hidden_size),
                "rgb_patch": nn.Linear(self.patch_feat_dim, self.hidden_size),
                "hand_rgb": nn.Linear(self.img_feat_dim, self.hidden_size),
                "hand_rgb_patch": nn.Linear(self.patch_feat_dim, self.hidden_size),
            }
        )

    def forward(self, batch):  # rgb, hand_rgb, state, lang):

        bs, seq, c, h, w = batch["rgb"].shape

        embeddings = {
            k: self.extractor[k](v) for k, v in batch.items() if v is not None
        }

        targets = {
            k: self.extractor[k].targets(v)
            for k, v in batch.items()
            if v is not None and k in self.visual_keys
        }

        for k in self.visual_keys:
            if k in embeddings:
                embeddings[f"{k}_patch"] = self.extractor[k].process_patch(
                    embeddings[k][1]
                )
                embeddings[k] = embeddings[k][0]

        for k in embeddings:
            if k in self.embed_fc and embeddings[k] is not None:
                embeddings[k] = self.embed_fc[k](embeddings[k])

        return embeddings, targets

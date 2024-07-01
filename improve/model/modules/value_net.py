import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many
from flamingo_pytorch import GatedCrossAttentionBlock, PerceiverResampler
from torch import einsum, nn


class QRNet(nn.Module):

    def __init__(
        self,
        num_quantiles=32,
        num_actions=10,
        act_dim=7,
        dim=192,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=64,
        num_media_embeds=4,
    ):
        super(QRNet, self).__init__()

        self.num_quantiles = num_quantiles
        self.num_actions = num_actions
        self.act_dim = act_dim
        self.dim = dim

        # self.perceiver = PerceiverResampler( dim=dim, depth=depth, dim_head=dim_head, heads=heads, num_latents=num_latents, num_media_embeds=num_media_embeds,)
        self.xattn = GatedCrossAttentionBlock(dim=dim, dim_head=64, heads=8)
        self.fc1 = nn.Linear(act_dim, dim)
        self.fc2 = nn.Linear(dim, num_quantiles)

    def forward(self, x, action=None):

        if len(action.shape) == 3:
            action = action.unsqueeze(-2)
        bs, seq, *other = x.shape

        assertion = "Action must be provided for multi-action envs"
        assert not (action is None and self.num_actions > 1), assertion

        x = rearrange(x, "b t n d -> (b t) 1 n d")

        bs, t, n, d = action.shape
        action = rearrange(action, "b t n d -> (b t) n d")

        action = self.fc1(action)

        quantiles = self.xattn(action, x)
        quantiles = rearrange(quantiles, "(b t) n d -> b t n d", b=bs, t=t)
        quantiles = F.sigmoid(self.fc2(quantiles))
        return quantiles

    def _predict(self, quants, actions=None):
        """predict the quantile values and the action
        Args:
            quants (torch.Tensor): quantile values (batch, seq, nact, nquant)
            actions (torch.Tensor): actions (batch, seq, nact, act_dim)
        Returns:
            quant (torch.Tensor): predicted quantile values (batch, seq, nact)
            action (torch.Tensor): predicted action (batch, seq, act_dim)
        """
        bs, seq, nact, nquant = quants.shape
        vals = quants.mean(dim=-1)

        # Greedy action
        idx = vals.argmax(dim=-1).view(bs, seq, 1, 1)

        # Gather the values from tgt using idx along the -2 dimension
        if actions is not None:
            action = (
                torch.gather(actions, 2, idx.expand(bs, seq, nact, actions.shape[-1]))
            )[:, :, 0, :]
        else:
            action = None

        quant = quants.gather(-2, idx.expand(bs, seq, nact, nquant))[:, :, 0, :]

        return quant, action

    def predict(self, x, actions):
        quants = self(x, actions)
        return self._predict(quants, actions)


def main():

    x = torch.randn(4, 10, 10, 192)
    action = torch.randn(4, 10, 10, 7)

    qrnet = QRNet()
    quantiles = qrnet(x, action)
    print(quantiles.shape)
    quant, act = qrnet._predict(quantiles, action)

    # print(quantiles[0, 0])
    # print(quant[0, 0])
    # print(act[0, 0])
    print(quant.shape, act.shape)

    quit()

    quantiles = rearrange(quantiles, "(b t) n d -> b t n d", b=4, t=10)
    print(quantiles.shape)
    quit()

    # Initialize PerceiverResampler
    perceiver_resampler = PerceiverResampler(
        dim=192,
        depth=2,
        dim_head=64,
        heads=8,
        num_latents=64,
        num_media_embeds=10,
    )

    resampled_latent = perceiver_resampler(x)
    print(resampled_latent.shape)  # Should output (4, 10, 10, 192)
    quit()

    cross_attention = GatedCrossAttentionBlock(
        dim=192, dim_head=64, heads=4, ff_mult=4, only_attend_immediate_media=False
    )

    merged_output = cross_attention(b, x)

    print(merged_output.shape)  # Should output (4, 10, 10, 7)

    """
    self.perceiver = PerceiverResampler(
        dim=patch_feat_dim,
        depth=resampler_params["depth"],
        dim_head=resampler_params["dim_head"],
        heads=resampler_params["heads"],
        num_latents=self.n_patch_latents,
        num_media_embeds=resampler_params["num_media_embeds"],
    )
    """


if __name__ == "__main__":
    main()

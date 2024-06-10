import math

import torch
import torch.nn.functional as F


def masked_loss(pred, target, mask, skip_frame=0, loss_func=F.mse_loss):

    # prepare skip frames
    pred = pred if skip_frame == 0 else pred[:, :-skip_frame]
    target = target[:, skip_frame:]
    mask = mask[:, skip_frame:]

    loss = loss_func(pred, target, reduction="none")

    for _ in range(pred.dim() - mask.dim()):
        mask = mask.unsqueeze(-1)
    loss = (loss * mask).sum() / mask.sum() / math.prod(pred.shape[mask.dim() :])
    return loss


def old_masked_loss(pred, target, mask, skip_frame=0, loss_func=F.mse_loss):

    if skip_frame == 0:
        new_pred = pred
    else:
        new_pred = pred[:, :-skip_frame]

    new_target = target[:, skip_frame:]
    new_mask = mask[:, skip_frame:]
    data_shape, mask_shape = new_pred.shape, new_mask.shape

    loss = loss_func(new_pred, new_target, reduction="none")

    for _ in range(len(data_shape) - len(mask_shape)):
        new_mask = new_mask.unsqueeze(-1)
    loss = (
        (loss * new_mask).sum()
        / new_mask.sum()
        / math.prod(data_shape[len(mask_shape) :])
    )
    return loss


def quantile_huber_loss(pred, target, cdf=None, reduction="none"):
    """
    The quantile-regression loss, as described in the QR-DQN and TQC papers.
    Partially taken from https://github.com/bayesgroup/tqc_pytorch.

    :param pred: current estimate of quantiles
    :param target: target of quantiles
    :param cdf: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper),
        (if None, calculating unit quantiles)
    :param reduction: if summing over the quantile dimension or not
    :return: the loss
    """

    bs, seq, bins = pred.shape

    # Cumulative probabilities to calculate quantiles.
    cdf = (torch.arange(bins, device=pred.device, dtype=torch.float) + 0.5) / bins
    cdf = cdf.view(1, 1, bins)

    # deltas are [bs,seq,bins,bins] because we sum over the bins dimension
    # because quantiles are CDF
    delta = target.unsqueeze(-2) - pred.unsqueeze(-1)
    absdelta = torch.abs(delta)

    # huber loss is most sensitive to non-outliers
    # minimize the squared delta for small errors and the absolute delta for large errors
    huber_loss = torch.where(absdelta > 1, absdelta - 0.5, delta**2 * 0.5)
    loss = torch.abs(cdf - (delta.detach() < 0).float()) * huber_loss
    loss = loss.sum(dim=-2)  # loss per bin

    reductions = {"none": loss, "mean": loss.mean()}
    return reductions[reduction]


""" with F.huber_loss
huber_loss = F.huber_loss(pairwise_delta_2d, abs_pairwise_delta_2d, reduction="none")
"""

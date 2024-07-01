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

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2 


def random_shift(x, pad):
    """Applies random shifts to the input tensor 'x' for data augmentation.

    Parameters:
    x (torch.Tensor): The input tensor of shape (b, n, c, w, h)
    pad (int): The padding size.

    Note:
    The function first pads the input tensor on all sides with 'pad' number of pixels. Then, it generates a random shift
    in the range of [-pad, pad] for each image in the batch. The shift is applied to the padded image, and the result is
    cropped to the original image size. This results in a randomly shifted version of the original image.
    """

    x = x.float()
    b, t, c, h, w = x.size()
    assert h == w
    x = x.view(b * t, c, h, w)  # reshape x to [B*T, C, H, W]
    padding = tuple([pad] * 4)
    x = F.pad(x, padding, "replicate")

    # calculate the height and width after padding
    h_pad, w_pad = h + 2 * pad, w + 2 * pad
    eps = 1.0 / (h_pad)

    arange = torch.linspace(
        -1.0 + eps, 1.0 - eps, h_pad, device=x.device, dtype=x.dtype
    )[:h]

    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(b * t, 1, 1, 1)

    shift = torch.randint(
        0, 2 * pad + 1, size=(b, 1, 1, 1, 2), device=x.device, dtype=x.dtype
    )

    # repeat the shift for each image in the sequence
    shift = shift.repeat(1, t, 1, 1, 1)
    # reshape shift to match the size of base_grid
    shift = shift.view(b * t, 1, 1, 2)
    shift *= 2.0 / (h_pad)

    grid = base_grid + shift
    output = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    output = output.view(b, t, c, h, w)  # reshape output back to [B, T, C, H, W]
    return output


class PreProcess:
    def __init__( self, cn, device):

        self.rgb_static_pad=cn.rgb_static_pad
        self.rgb_gripper_pad=cn.rgb_gripper_pad

        self.rgb_shape=cn.rgb_shape
        self.resize = v2.Resize(self.rgb_shape, interpolation=Image.BICUBIC, antialias=True).to(
            device
        )
        self.rgb_mean = torch.tensor(cn.rgb_mean, device=device).view(1, 1, -1, 1, 1)
        self.rgb_std = torch.tensor(cn.rgb_std, device=device).view(1, 1, -1, 1, 1)

    def rgb_process(self, rgb_static, rgb_gripper, train=False):

        rgb_static = rgb_static.float() * (1 / 255.0)
        rgb_gripper = rgb_gripper.float() * (1 / 255.0)

        if train:
            rgb_static = random_shift(rgb_static, self.rgb_static_pad)
            rgb_gripper = random_shift(rgb_gripper, self.rgb_gripper_pad)

        rgb_static = self.resize(rgb_static)
        rgb_gripper = self.resize(rgb_gripper)
        # torchvision Normalize forces sync between CPU and GPU, so we use our own
        rgb_static = (rgb_static - self.rgb_mean) / (self.rgb_std + 1e-6)
        rgb_gripper = (rgb_gripper - self.rgb_mean) / (self.rgb_std + 1e-6)
        return rgb_static, rgb_gripper

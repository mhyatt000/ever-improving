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

    if h != w:
        x = crop_square(x)
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


def crop_square(x):
    b, t, c, h, w = x.size()
    dim = min(h, w)

    # Calculate the starting points for cropping
    h = (h - dim) // 2
    w = (w - dim) // 2

    # Crop the x
    cropped = x[:, :, :, h : h + dim, w : w + dim]
    return cropped

def to_channels_first(image):
    return image.permute(0, 1, 4, 2, 3)

def is_channels_first(image):
    """ Check if the image tensor is in channels first format.
    -3 works for both 4D and 5D tensors.
    """
    return image.size(-3) == 3

class PreProcess:
    def __init__(self, cn, device):

        self.rgb_static_pad = cn.rgb_static_pad
        self.rgb_gripper_pad = cn.rgb_gripper_pad

        self.rgb_shape = cn.rgb_shape
        self.resize = v2.Resize(
            self.rgb_shape, interpolation=Image.BICUBIC, antialias=True
        ).to(device)
        self.rgb_mean = torch.tensor(cn.rgb_mean, device=device).view(1, 1, -1, 1, 1)
        self.rgb_std = torch.tensor(cn.rgb_std, device=device).view(1, 1, -1, 1, 1)

    def _process(self, image, static, train=False):
        pad = self.rgb_static_pad if static else self.rgb_gripper_pad
        image = image.float() * (1 / 255.0)

        if not is_channels_first(image):
            image = to_channels_first(image)

        if train:
            image = random_shift(image, pad)

        image = self.resize(image)
        # torchvision Normalize forces sync between CPU and GPU, so we use our own
        image = (image - self.rgb_mean) / (self.rgb_std + 1e-6)

        return image

    def rgb_process(self, rgb_static, rgb_gripper, train=False):
        rgb_static = self._process(rgb_static, True, train)
        rgb_gripper = self._process(rgb_gripper, False, train)

        return rgb_static, rgb_gripper

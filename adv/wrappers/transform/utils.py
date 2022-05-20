import math
from typing import Union

import numpy as np
import torch
from kornia.color import xyz_to_rgb
from kornia.enhance.adjust import adjust_saturation_raw
from torch._vmap_internals import vmap

EPS = 1e-9


def to_numpy(x):
    return x.permute(1, 2, 0).detach().cpu().numpy()


def from_numpy(x):
    return torch.from_numpy(x).permute(2, 0, 1)


def diff_round(x):
    """Differentiable rounding."""
    return torch.round(x) + (x - torch.round(x)) ** 3


def diff_remainder(x):
    # If x is round down we are good. Otherwise, we have to invert it.
    # Gradient of this is 1 - 3x^2 for x in [-0.5, 0.5] and repeat.
    # The gradient is also continuous but not smooth.
    y = x - diff_round(x)
    return torch.where(y < 0, 1 + y, y)


def batchify(transform_single):
    def transform(x, hparams, same_on_batch=False):
        x_new = torch.zeros_like(x)
        params = []
        for i, im in enumerate(x):
            im_new, im_params = transform_single(im, hparams)
            x_new[i] = im_new
            params.append(im_params)
        if params[0] is None:
            params = None
        else:
            params = torch.from_numpy(np.array(params)).to(x.device)
        x_new = x_new.to(x.device)
        return x_new, params
    return transform


def batchify_vector(transform_single):
    def transform(x, hparams, same_on_batch=False):
        transform_batch = vmap(lambda im: transform_single(im, hparams)[0])
        x_new = transform_batch(x)
        return x_new, None
    return transform


def normalize(val, min_val, max_val):
    if min_val == max_val:
        return 0 * val
    return (val - min_val) / (max_val - min_val)


def samplewise_normalize(x):
    """
    Normalize each sample in the batch such that its min and max valuese are
    zero and one.
    """
    x_max = x.view(x.size(0), -1).max(1)[0][(..., ) + (None, ) * (x.dim() - 1)]
    x_min = x.view(x.size(0), -1).min(1)[0][(..., ) + (None, ) * (x.dim() - 1)]
    # NOTE: if x == x_min == x_max then, output is all 0.5
    return (x - x_min + 0.5 * 1e-6) / (x_max - x_min + 1e-6)


def get_mask(x, mode='random'):
    """
    Return zero mask of an appropriate size.
    Assume that `x` has 4 dims: (B, C, H, W).
    """
    assert x.ndim == 4
    if mode == 'random':
        # Random mode only includes 'all' and 'pixelwise' masks
        mode = np.random.choice(['all', 'pixelwise'])
    size = list(x.size())
    if mode == 'all':
        pass
    elif mode == 'channelwise':
        # output is (B, C, 1, 1)
        size[2] = 1
        size[3] = 1
    elif mode == 'pixelwise':
        # output is (B, 1, H, W)
        size[1] = 1
    elif mode == 'samplewise':
        # output is (B, 1, 1, 1)
        size[1] = 1
        size[2] = 1
        size[3] = 1
    else:
        raise ValueError('Invalid get_mask() mode!')
    return torch.zeros(size, device=x.device, dtype=x.dtype)


def lab_to_rgb(image: torch.Tensor, clip: bool = True) -> torch.Tensor:
    r"""Converts a Lab image to RGB.

    Args:
        image (torch.Tensor): Lab image to be converted to RGB with shape :math:`(*, 3, H, W)`.
        clip (bool): Whether to apply clipping to insure output RGB values in range :math:`[0, 1]`. Default is True

    Returns:
        torch.Tensor: Lab version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = lab_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    L: torch.Tensor = image[..., 0, :, :]
    a: torch.Tensor = image[..., 1, :, :]
    _b: torch.Tensor = image[..., 2, :, :]

    fy = (L + 16.) / 116.
    fx = (a / 500.) + fy
    fz = fy - (_b / 200.)

    # if color data out of range: Z < 0
    fz = torch.where(fz < 0, torch.zeros_like(fz), fz)

    fxyz = torch.stack([fx, fy, fz], dim=-3)

    # Convert from Lab to XYZ
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4. / 29.) / 7.787
    xyz = torch.where(fxyz > .2068966, power, scale)

    # For D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1., 1.08883], device=xyz.device, dtype=xyz.dtype)[..., :, None, None]
    xyz_im = xyz * xyz_ref_white

    rgbs_im: torch.Tensor = xyz_to_rgb(xyz_im).clamp_min(EPS)

    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
    # rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)

    # Convert from sRGB to RGB Linear
    rs: torch.Tensor = rgbs_im[..., 0, :, :]
    gs: torch.Tensor = rgbs_im[..., 1, :, :]
    bs: torch.Tensor = rgbs_im[..., 2, :, :]

    r: torch.Tensor = torch.where(
        rs > 0.0031308, 1.055 * torch.pow(rs, 1 / 2.4) - 0.055, 12.92 * rs)
    g: torch.Tensor = torch.where(
        gs > 0.0031308, 1.055 * torch.pow(gs, 1 / 2.4) - 0.055, 12.92 * gs)
    b: torch.Tensor = torch.where(
        bs > 0.0031308, 1.055 * torch.pow(bs, 1 / 2.4) - 0.055, 12.92 * bs)

    rgb_im: torch.Tensor = torch.stack([r, g, b], dim=-3)

    # Clip to 0,1 https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)

    return rgb_im


def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: HSV version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    # The first or last occurance is not guarenteed before 1.6.0
    # https://github.com/pytorch/pytorch/issues/20414
    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / v.clamp_min(EPS)

    # avoid division by zero
    deltac = torch.where(deltac < EPS, torch.ones_like(
        deltac, device=deltac.device, dtype=deltac.dtype), deltac)

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: torch.Tensor = maxc_tmp[..., 0, :, :]
    gc: torch.Tensor = maxc_tmp[..., 1, :, :]
    bc: torch.Tensor = maxc_tmp[..., 2, :, :]

    h = torch.stack([
        bc - gc,
        2.0 * deltac + rc - bc,
        4.0 * deltac + gc - rc,
    ], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac

    # NOTE: Replace this line with a differentiable version
    # h = (h / 6.0) % 1.0
    h = diff_remainder(h / 6)

    h = 2 * math.pi * h

    return torch.stack([h, s, v], dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    # `hi` is only used as indexing so it should be ok
    hi: torch.Tensor = torch.floor(h * 6) % 6

    # NOTE: Change this line to a differentiable version
    # f: torch.Tensor = ((h * 6) % 6) - hi
    f: torch.Tensor = diff_remainder(h) * 6 - hi

    # one: torch.Tensor = torch.tensor(1.).to(image.device)
    # p: torch.Tensor = v * (one - s)
    # q: torch.Tensor = v * (one - f * s)
    # t: torch.Tensor = v * (one - (one - f) * s)
    p: torch.Tensor = v * (1 - s)
    q: torch.Tensor = v * (1 - f * s)
    t: torch.Tensor = v * (1 - (1 - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((
        v, q, p, p, t, v,
        t, v, v, q, p, p,
        p, p, t, v, v, q,
    ), dim=-3)
    out = torch.gather(out, -3, indices)

    return out


def adjust_saturation(input: torch.Tensor, saturation_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust color saturation of an image.

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        input (torch.Tensor): Image/Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
        saturation_factor (Union[float, torch.Tensor]):  How much to adjust the saturation. 0 will give a black
          and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.

    Return:
        torch.Tensor: Adjusted image in the shape of :math:`(*, 3, H, W)`.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> adjust_saturation(x, 2.)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.ones(2)
        >>> out = adjust_saturation(x, y)
        >>> torch.nn.functional.mse_loss(x, out)
        tensor(0.)
    """

    # convert the rgb image to hsv
    x_hsv: torch.Tensor = rgb_to_hsv(input)

    # perform the conversion
    x_adjusted: torch.Tensor = adjust_saturation_raw(x_hsv, saturation_factor)

    # convert back to rgb
    out: torch.Tensor = hsv_to_rgb(x_adjusted)

    return out


def adjust_hue(input: torch.Tensor, hue_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust hue of an image.

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        input (torch.Tensor): Image to be adjusted in the shape of :math:`(*, 3, H, W)`.
        hue_factor (Union[float, torch.Tensor]): How much to shift the hue channel. Should be in [-PI, PI]. PI
          and -PI give complete reversal of hue channel in HSV space in positive and negative
          direction respectively. 0 means no shift. Therefore, both -PI and PI will give an
          image with complementary colors while 0 gives the original image.

    Return:
        torch.Tensor: Adjusted image in the shape of :math:`(*, 3, H, W)`.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> adjust_hue(x, 3.141516)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.ones(2) * 3.141516
        >>> adjust_hue(x, y).shape
        torch.Size([2, 3, 3, 3])
    """

    # convert the rgb image to hsv
    x_hsv: torch.Tensor = rgb_to_hsv(input)

    # perform the conversion
    x_adjusted: torch.Tensor = adjust_hue_raw(x_hsv, hue_factor)

    # convert back to rgb
    out: torch.Tensor = hsv_to_rgb(x_adjusted)

    return out


def adjust_hue_raw(input: torch.Tensor, hue_factor: Union[float, torch.Tensor]) -> torch.Tensor:
    r"""Adjust hue of an image. Expecting input to be in hsv format already.
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(hue_factor, (float, torch.Tensor)):
        raise TypeError(f"The hue_factor should be a float number or torch.Tensor in the range between"
                        f" [-PI, PI]. Got {type(hue_factor)}")

    if isinstance(hue_factor, float):
        hue_factor = torch.as_tensor(hue_factor)

    hue_factor = hue_factor.to(input.device, input.dtype)

    # TODO: find a proper way to check bound values in batched tensors.
    # if ((hue_factor < -pi) | (hue_factor > pi)).any():
    #     raise ValueError(f"Hue-factor must be in the range [-PI, PI]. Got {hue_factor}")

    for _ in input.shape[1:]:
        hue_factor = torch.unsqueeze(hue_factor, dim=-1)

    # unpack the hsv values
    h, s, v = torch.chunk(input, chunks=3, dim=-3)

    # transform the hue value and apply module
    divisor: float = 2 * math.pi
    # Replace this line with differentiable version
    # h_out: torch.Tensor = torch.fmod(h + hue_factor, divisor)
    h_out: torch.Tensor = diff_remainder((h + hue_factor) / divisor) * divisor

    # pack back back the corrected hue
    out: torch.Tensor = torch.cat([h_out, s, v], dim=-3)

    return out

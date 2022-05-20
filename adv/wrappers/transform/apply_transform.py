'''
Implement functions for applying transformations.
Currently, all transforms are not in-place.
TODO: Can we improve speed by optionally changing them to in-place?
'''
import math
import os
import random
from copy import deepcopy
from io import BytesIO

import kornia
import numpy as np
import PIL
import scipy
import skimage
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
import yaml
from kornia.enhance.adjust import (adjust_brightness, adjust_contrast,
                                   adjust_gamma, sharpness)
from skimage.morphology import disk
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import swirl as skswirl
from torch.fft import irfftn, rfftn

# TODO: consider fixing relative imports
from ...models.bpda import Identity, get_bpda_network
from .init_transform import init_transform, name_to_init_tf
from .utils import (EPS, adjust_hue, adjust_saturation, batchify,
                    batchify_vector, diff_round, from_numpy, get_mask,
                    hsv_to_rgb, lab_to_rgb, normalize, rgb_to_hsv,
                    samplewise_normalize, to_numpy)


def apply_module(x, hparams, module, same_on_batch=False):
    return module(x), None


def apply_module_normalize(x, hparams, module, same_on_batch=False):
    x_new, params = apply_module(x, hparams, module)
    return samplewise_normalize(x_new), params


def transform_affine(x, hparams, same_on_batch=False):
    affine_theta = torch.eye(3, device=x.device).repeat(x.shape[0], 1, 1)
    alpha = hparams['alpha']
    distr = hparams.get('dist', 'uniform')
    perturb = torch.zeros_like(affine_theta)
    if distr == 'uniform':
        perturb.uniform_(- alpha, alpha)
    elif distr == 'normal':
        perturb.normal_(0, alpha)
    if same_on_batch:
        affine_theta += perturb[0:1]
    else:
        affine_theta += perturb
    grid = F.affine_grid(affine_theta[:, :2, :], x.size())
    x_new = F.grid_sample(x, grid)
    # For rand_wrapper_v2, we just take average of `perturb`
    return x_new, perturb.mean((1, 2))


def drop_pixel(x, hparams, same_on_batch=False):
    # This the same as pepper noise
    drop_rate = random.uniform(0., hparams['alpha'])
    mask = get_mask(x, mode='random')
    mask.bernoulli_(drop_rate)
    if same_on_batch:
        mask = mask[0:1]
    x_new = x - mask
    return x_new, None


def add_salt_noise(x, hparams, same_on_batch=False):
    drop_rate = random.uniform(0., hparams['alpha'])
    mask = get_mask(x, mode='random')
    mask.bernoulli_(drop_rate)
    if same_on_batch:
        mask = mask[0:1]
    x_new = x + mask
    return x_new, None


def jitter_color(x, hparams, same_on_batch=False):
    alpha = hparams['alpha']
    num_total = x.size(0)
    device = x.device
    distr = hparams.get('hist', 'uniform')
    if distr == 'uniform':
        brightness = torch.zeros(num_total, device=device).uniform_(- alpha, alpha)
        contrast = torch.zeros(num_total, device=device).uniform_(1 - alpha, 1 + alpha)
        saturation = torch.zeros(num_total, device=device).uniform_(1 - alpha, 1 + alpha)
        hue = torch.zeros(num_total, device=device).uniform_(- alpha, alpha)
    elif distr == 'normal':
        brightness = torch.zeros(num_total, device=device).normal_(0, alpha)
        contrast = torch.zeros(num_total, device=device).normal_(1, alpha)
        saturation = torch.zeros(num_total, device=device).normal_(1, alpha)
        hue = torch.zeros(num_total, device=device).normal_(0, alpha)
    else:
        raise NotImplementedError('No specified distribution!')

    if same_on_batch:
        brightness = brightness[0:1].expand(num_total).clone()
        contrast = contrast[0:1].expand(num_total).clone()
        saturation = saturation[0:1].expand(num_total).clone()
        hue = hue[0:1].expand(num_total).clone()

    # Bound parameters to valid range
    # NOTE: Kornia api is not very stable. Double check with the
    # latest code for `adjust_XXX()`.
    brightness.clamp_(-1, 1)
    contrast.clamp_min_(EPS)
    saturation.clamp_min_(EPS)
    hue.clamp_(-math.pi + EPS, math.pi - EPS)

    # Clipping is needed to prevent NaN gradient during attack
    x_new = x.clone().clamp(EPS, 1)
    for i in np.random.permutation(4):
        if i == 0:
            x_new = adjust_brightness(x_new, brightness)
            # NOTE: NaN input can happen if all pixels have the same
            # value which can be caused by brightness adjustment.
            # This small noise injection mitigates the problem.
            if same_on_batch:
                x_new += torch.zeros_like(x[0:1]).uniform_(-1e-5, 1e-5)
            else:
                x_new += torch.zeros_like(x_new).uniform_(-1e-5, 1e-5)
            x_new.clamp_(1e-5, 1 - 1e-5)
        elif i == 1:
            x_new = adjust_contrast(x_new, contrast)
        elif i == 2:
            x_new = adjust_saturation(x_new, saturation)
        else:
            x_new = adjust_hue(x_new, hue)
        x_new.clamp_(EPS, 1)

    # For rand_wrapper_v2, we just take average of the four params
    return x_new, (brightness.abs() + contrast + saturation + hue.abs()) / 4


def alter_gamma(x, hparams, same_on_batch=False):
    gamma = torch.zeros(x.size(0), device=x.device)
    distr = hparams.get('dist', 'uniform')
    if distr == 'uniform':
        gamma.uniform_(1 - hparams['alpha'], 1 + hparams['alpha'])
    elif distr == 'normal':
        gamma.normal_(1, hparams['alpha'])
    gamma.clamp_min_(EPS)
    if same_on_batch:
        gamma = gamma[0:1].expand(x.size(0))
    x_new = adjust_gamma(torch.clamp_min(x, EPS), gamma)
    return x_new, gamma


def add_gaussian_noise(x, hparams, same_on_batch=False):
    if 'std' in hparams:
        std = hparams['std']
    else:
        std = hparams['alpha']
    mask = get_mask(x, mode='random')
    mask.normal_(hparams.get('mean', 0.), std)
    if same_on_batch:
        mask = mask[0:1]
    x_new = x + mask
    return x_new, None


def add_poisson_noise(x, hparams, same_on_batch=False):
    noisiness = hparams['alpha']
    noise = torch.poisson(x / noisiness) * noisiness - x.clone().detach()
    if same_on_batch:
        # Since Poisson rate depends on the input, we have to fix the input to
        # 1 to make the noise same across all batches
        noise = torch.poisson(torch.ones_like(x[0:1]) / noisiness) * noisiness - 1
    x_new = x + noise
    return x_new, None


def add_speckle_noise(x, hparams, same_on_batch=False):
    mask = get_mask(x, mode='random')
    mask.normal_(hparams.get('mean', 1.), hparams['alpha'])
    if same_on_batch:
        mask = mask[0:1]
    x_new = x * mask
    return x_new, None


def add_uniform_noise(x, hparams, same_on_batch=False):
    if 'range' in hparams:
        lo, hi = hparams['range']
    else:
        lo, hi = - hparams['alpha'], hparams['alpha']
    mask = get_mask(x, mode='random')
    mask.uniform_(lo, hi)
    if same_on_batch:
        mask = mask[0:1]
    x_new = x + mask
    return x_new, None


def fft(x, hparams, same_on_batch=False):
    hparams['min_pf'], hparams['max_pf'] = 0.98, 1.02
    hparams['min_frac'], hparams['max_frac'] = EPS, hparams['alpha']
    return fft_full(x, hparams, same_on_batch=same_on_batch)


def fft_full(x, hparams, same_on_batch=False):
    size = x.shape[0]
    min_pf, max_pf = hparams['min_pf'], hparams['max_pf']
    min_frac, max_frac = hparams['min_frac'], hparams['max_frac']
    x_freq = rfftn(x, dim=(2, 3))
    # Scales all channels by a random constant
    scale = get_mask(x, mode='samplewise')
    scale.uniform_(min_pf, max_pf)
    if same_on_batch:
        scale = scale[0:1]
    x_freq *= scale
    # Randomly drop some components
    drop_rate = random.uniform(min_frac, max_frac)
    dr_expanded = drop_rate * torch.ones(size, 1)
    params = torch.cat((scale.view(size, 1), dr_expanded.to(scale.device)), dim=1)
    mask = get_mask(torch.real(x_freq), mode='random')
    mask = mask.bernoulli_(1 - drop_rate)
    if same_on_batch:
        mask = mask[0:1]
    x_freq *= mask
    x_new = irfftn(x_freq, dim=(2, 3))
    return x_new, params


def jpeg_compress(x, hparams, module, same_on_batch=False):
    alpha = hparams['alpha']
    hparams['min'], hparams['max'] = 5, (1 - alpha) * 80 + 5
    return jpeg_compress_full(x, hparams, module, same_on_batch=same_on_batch)


def jpeg_compress_full(x, hparams, module, same_on_batch=False):
    min_qual, max_qual = hparams['min'], hparams['max']
    quality = torch.zeros(x.size(0), device=x.device)
    distr = hparams.get('dist', 'uniform')
    if distr == 'uniform':
        # quality of 100 is original: [5, 99]
        quality.uniform_(min_qual, max_qual)
    elif distr == 'normal':
        quality.normal_(hparams['mean'], hparams['std'])
        quality = torch.clamp(min_qual, max_qual)
    params = normalize(quality.unsqueeze(1), min_qual, max_qual)
    x_new = module(x, quality=torch.round(quality))
    return x_new, params


def reduce_color_precision_diff(x, hparams, same_on_batch=False):
    # Min precision: 0 bit, max precision: 4 bits
    hparams['min'] = int(16 * (1 - hparams['alpha'])) + 1
    hparams['max'] = 16
    return reduce_color_precision_diff_full(x, hparams, same_on_batch=same_on_batch)


def reduce_color_precision_diff_full(x, hparams, same_on_batch=False):
    size, n_ch, _, _ = x.shape
    if same_on_batch:
        size = 1
    min_val, max_val = hparams['min'], hparams['max']
    scales = torch.randint(min_val, max_val + 1, (size, n_ch), device=x.device)
    scales_dim = scales.view(size, n_ch, 1, 1)
    x_new = x * scales_dim
    x_new = diff_round(x_new) / scales_dim
    params = normalize(scales, min_val, max_val)
    return x_new, params


def alter_sharpness(x, hparams, same_on_batch=False):
    distr = hparams.get('dist', 'uniform')
    sharp = torch.zeros(x.size(0), device=x.device)
    if distr == 'uniform':
        sharp.uniform_(- hparams['alpha'], hparams['alpha'])
    elif distr == 'normal':
        sharp.normal_(0, hparams['alpha'])
    if same_on_batch:
        sharp = sharp[0:1].expand(x.size(0))
    x_new = sharpness(x, 10 ** (sharp * 10))
    return x_new, sharp


def alter_xyz(x, hparams, same_on_batch=False):
    hparams['min'] = - hparams['alpha']
    hparams['max'] = hparams['alpha']
    return alter_xyz_full(x, hparams, same_on_batch=same_on_batch)


def alter_xyz_full(x, hparams, same_on_batch=False):
    min_pert, max_pert = hparams['min'], hparams['max']
    x_new = kornia.color.rgb_to_xyz(x)
    perturb = get_mask(x, mode='channelwise')
    perturb.uniform_(min_pert, max_pert)
    if same_on_batch:
        perturb = perturb[0:1]
    x_new += perturb
    x_new.clamp_(0, 1)
    x_new = kornia.color.xyz_to_rgb(x_new)
    params = normalize(perturb, min_pert, max_pert)
    return x_new, params


def alter_yuv(x, hparams, same_on_batch=False):
    alpha = hparams['alpha']
    hparams['min_y'], hparams['max_y'] = - alpha, alpha
    hparams['min_uv'], hparams['max_uv'] = - alpha * 0.4, alpha * 0.4
    return alter_yuv_full(x, hparams, same_on_batch=same_on_batch)


def alter_yuv_full(x, hparams, same_on_batch=False):
    min_y, max_y = hparams['min_y'], hparams['max_y']
    min_uv, max_uv = hparams['min_uv'], hparams['max_uv']
    x_new = kornia.color.rgb_to_yuv(x)
    perturb = get_mask(x, mode='channelwise')
    perturb[:, 0].uniform_(min_y, max_y)
    perturb[:, 1:].uniform_(min_uv, max_uv)
    if same_on_batch:
        perturb = perturb[0:1]
    x_new += perturb
    x_new.clamp_(0, 1)
    x_new = kornia.color.yuv_to_rgb(x_new)
    y_pert_norm = normalize(perturb[:, 0].unsqueeze(-1), min_y, max_y)
    uv_pert_norm = normalize(perturb[:, 1:], min_uv, max_uv)
    params = torch.cat((y_pert_norm, uv_pert_norm), dim=1)
    return x_new, params


def alter_lab(x, hparams, same_on_batch=False):
    alpha = hparams['alpha']
    hparams['min_l'], hparams['max_l'] = - alpha * 100, alpha * 100
    hparams['min_ab'], hparams['max_ab'] = - alpha * 40, alpha * 40
    return alter_lab_full(x, hparams, same_on_batch=same_on_batch)


def alter_lab_full(x, hparams, same_on_batch=False):
    min_l, max_l = hparams['min_l'], hparams['max_l']
    min_ab, max_ab = hparams['min_ab'], hparams['max_ab']
    x_new = kornia.color.rgb_to_lab(x.clamp_min(EPS))
    perturb = get_mask(x, mode='channelwise')
    perturb[:, 0].uniform_(min_l, max_l)
    perturb[:, 1:].uniform_(min_ab, max_ab)
    if same_on_batch:
        perturb = perturb[0:1]
    x_new += perturb
    x_new.clamp_(0, 100)
    x_new = lab_to_rgb(x_new)
    l_pert_norm = normalize(perturb[:, 0].unsqueeze(-1), min_l, max_l)
    ab_pert_norm = normalize(perturb[:, 1:], min_ab, max_ab)
    params = torch.cat((l_pert_norm, ab_pert_norm), dim=1)
    return x_new, params


def alter_hsv(x, hparams, same_on_batch=False):
    alpha = hparams['alpha']
    hparams['min_h'], hparams['max_h'] = - alpha / 5, alpha / 5
    hparams['min_sv'], hparams['max_sv'] = - alpha, alpha
    return alter_hsv_diff_full(x, hparams, same_on_batch=same_on_batch)


def alter_hsv_full(x, hparams, same_on_batch=False):
    min_h, max_h = hparams['min_h'], hparams['max_h']
    min_sv, max_sv = hparams['min_sv'], hparams['max_sv']
    perturb = get_mask(x, mode='channelwise')
    perturb[:, 0].uniform_(min_h, max_h)
    perturb[:, 1:].uniform_(min_sv, max_sv)
    x_new = kornia.color.rgb_to_hsv(x)
    if same_on_batch:
        perturb = perturb[0:1]
    x_new += perturb
    x_new.clamp_(0, 1)
    x_new = kornia.color.hsv_to_rgb(x_new)
    h_pert_norm = normalize(perturb[:, 0].unsqueeze(-1), min_h, max_h)
    sv_pert_norm = normalize(perturb[:, 1:], min_sv, max_sv)
    params = torch.cat((h_pert_norm, sv_pert_norm), dim=1)
    return x_new, params


def alter_hsv_diff_full(x, hparams, same_on_batch=False):
    min_h, max_h = hparams['min_h'], hparams['max_h']
    min_sv, max_sv = hparams['min_sv'], hparams['max_sv']
    perturb = get_mask(x, mode='channelwise')
    perturb[:, 0].uniform_(min_h, max_h)
    perturb[:, 1:].uniform_(min_sv, max_sv)
    x_new = rgb_to_hsv(x)
    if same_on_batch:
        perturb = perturb[0:1]
    x_new += perturb
    x_new.clamp_(0, 1)
    x_new = hsv_to_rgb(x_new)
    h_pert_norm = normalize(perturb[:, 0].unsqueeze(-1), min_h, max_h)
    sv_pert_norm = normalize(perturb[:, 1:], min_sv, max_sv)
    params = torch.cat((h_pert_norm, sv_pert_norm), dim=1)
    return x_new, params


def mix_gray_scale(x, hparams, same_on_batch=False):
    assert not same_on_batch
    # Grayscale with random weights of RGB channels
    ratios = get_mask(x, mode='channelwise')
    ratios.uniform_(0., 1.)
    ratios /= ratios.sum(1, keepdim=True)
    x_new = (x * ratios).sum(1, keepdim=True).expand_as(x).clone()
    params = ratios.view(x.size(0), -1)
    return x_new, params


def mix_partial_gray_scale(x, hparams, same_on_batch=False):
    assert not same_on_batch
    # Grayscale that is randomly weighted by the original image
    gray, gray_params = mix_gray_scale(x, hparams)
    prop_ratios = get_mask(x, mode='channelwise')
    prop_ratios.uniform_(0., hparams['alpha'])
    x_new = x * (1 - prop_ratios) + gray * prop_ratios
    params = torch.cat((gray_params, prop_ratios.view(x.size(0), -1)), dim=1)
    return x_new, params


def mix_two_thirds_gray_scale(x, hparams, same_on_batch=False):
    assert not same_on_batch
    # Turn two of the three RGB channels to grayscale
    size, n_ch, _, _ = x.shape
    x_new, gray_params = mix_gray_scale(x, hparams)
    ch = torch.randint(n_ch, [size], device=x.device)
    x_new[torch.arange(size), ch, :, :] = x[torch.arange(size), ch, :, :]
    params = torch.cat(
        (gray_params, normalize(ch.unsqueeze(-1), 0, n_ch - 1)), dim=1)
    return x_new, params


def mix_one_partial_gray_scale(x, hparams, same_on_batch=False):
    assert not same_on_batch
    # Turn one of the three RGB channels to grayscale
    size, n_ch, _, _ = x.shape
    gray, gray_params = mix_gray_scale(x, hparams)
    ch = torch.randint(n_ch, [size], device=x.device)
    x_new = x.clone()
    x_new[torch.arange(size), ch, :, :] = gray[torch.arange(size), ch, :, :]
    params = torch.cat(
        (gray_params, normalize(ch.unsqueeze(-1), 0, n_ch - 1)), dim=1)
    return x_new, params


def _get_rand_ks(hparams, height):
    if 'min_ks' in hparams and 'max_ks' in hparams:
        min_ks, max_ks = hparams['min_ks'], hparams['max_ks']
        kernel_size = tuple(x.item() for x in 2 * np.random.randint(
            min_ks / 2, (max_ks + 1) / 2, size=2) + 1)
        params = normalize(np.array(kernel_size), min_ks, max_ks)
    else:
        # Hard-code min/max kernel size here
        if height < 100:
            min_ks, max_ks = 1, 3
        else:
            min_ks, max_ks = 3, 7
        kernel_size = tuple(x.item() for x in 2 * np.random.randint(
            min_ks, max_ks + 1, size=2) + 1)
        params = normalize(np.array(kernel_size), 2 * min_ks + 1, 2 * max_ks + 1)
    return kernel_size, params


def blur_gaussian_single(im, hparams):
    min_std, max_std = hparams.get('min_std', 1.), hparams.get('max_std', 5.)
    sigma = tuple(np.random.uniform(min_std, max_std, size=2))
    kernel_size, ks_params = _get_rand_ks(hparams, im.size(-1))
    im_new = kornia.filters.gaussian_blur2d(
        im.unsqueeze(0), kernel_size=kernel_size, sigma=sigma)
    im_new = im_new.squeeze(0)
    params = np.concatenate((normalize(np.array(sigma), min_std, max_std), ks_params))
    return im_new, params


def blur_mean_single(im, hparams):
    kernel_size, params = _get_rand_ks(hparams, im.size(-1))
    im_new = kornia.filters.box_blur(im.unsqueeze(0), kernel_size=kernel_size)
    im_new = im_new.squeeze(0)
    return im_new, params


def blur_median_single(im, hparams):
    kernel_size, params = _get_rand_ks(hparams, im.size(-1))
    im_new = kornia.filters.median_blur(
        im.unsqueeze(0), kernel_size=kernel_size)
    im_new = im_new.squeeze(0)
    return im_new, params


def get_gauss_kernel(ks, std):
    ax = torch.arange(-ks // 2 + 1., ks // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * std ** 2))
    return kernel


def shift(x, ks):
    n_ch, _, h, w = x.size()
    pad = ks // 2
    x_pad = F.pad(x, (pad, pad, pad, pad))
    # Alias for convenience
    cat_layers = []
    for i in range(n_ch):
        # Parse in row-major
        for y in range(ks):
            y2 = y + h
            for x in range(ks):
                x2 = x + w
                xx = x_pad[:, i:i + 1, y:y2, x:x2]
                cat_layers += [xx]
    return torch.cat(cat_layers, 1)


def blur_mean_bilateral_non_rand(im, ks, std_space, std_color):
    _, n_ch, height, width = im.shape
    std_color = 2 * std_color ** 2
    im_s = shift(im, ks)
    im_ex = im.expand(*im_s.size())
    d = (im_s - im_ex) ** 2
    d_e = torch.exp(-d / std_color)
    gw = get_gauss_kernel(ks, std_space)
    g = gw.view(n_ch, ks * ks, 1, 1).expand(-1, -1, height, width)
    d_d = d_e * g.to(im.device)
    w_denom = torch.sum(d_d, dim=1)
    im_f = torch.sum(d_d * im_s, dim=1) / w_denom
    return im_f


def blur_mean_bilateral_single(im, hparams, same_on_batch=False):
    n_ch, height, width = im.shape
    min_ks, max_ks = hparams.get('min_ks', 7), hparams.get('max_ks', 15)
    min_std_space = hparams.get('min_std_space', 1.)
    max_std_space = hparams.get('max_std_space', 5.)
    min_std_color = hparams.get('min_std_color', EPS)
    max_std_color = hparams.get('max_std_color', 0.3)
    ks = 2 * np.random.randint(min_ks / 2, (max_ks + 1) / 2, size=n_ch) + 1
    std_space = np.random.uniform(min_std_space, max_std_space, size=n_ch)
    std_color = np.random.uniform(min_std_color, max_std_color, size=n_ch)

    im_new = torch.zeros(im.shape)
    # TODO: Can we make this more elegant and potentially faster?
    for i in range(n_ch):
        im_new[i, :, :] = blur_mean_bilateral_non_rand(im[i, :, :].view(
            1, 1, height, width), ks[i], std_space[i], std_color[i]).squeeze()
    params = np.concatenate((normalize(ks, min_ks, max_ks),
                             normalize(std_space, min_std_space,
                                       max_std_space),
                             normalize(std_color, min_std_color, max_std_color)))
    return im_new, params


def map_swirl(coords, center, rotation, strength, radius):
    size, _, n_coords = coords.shape
    x, y = coords.permute(1, 0, 2).reshape(2, -1)
    center = center.unsqueeze(2).expand(size, 2, n_coords).permute(1, 0, 2).reshape(2, -1)
    x0, y0 = center
    rho = torch.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    # Ensure that the transformation decays to approximately 1/1000-th
    # within the specified radius.
    radius = radius / 5 * np.log(2)

    radius = radius.unsqueeze(1).expand(size, n_coords).flatten()
    strength = strength.unsqueeze(1).expand(size, n_coords).flatten()
    rotation = rotation.unsqueeze(1).expand(size, n_coords).flatten()

    theta = rotation + strength * torch.exp(-rho / radius) + torch.atan2(y - y0, x - x0)

    coords_new = torch.stack((x0 + rho * torch.cos(theta), y0 + rho * torch.sin(theta)))
    coords_new = coords_new.reshape(2, size, n_coords).permute(1, 0, 2)

    return coords_new


def bilinear_interpolate(im, coords):
    size, n_ch, height, width = im.shape
    n_coords = coords.shape[2]

    x, y = coords.permute(1, 0, 2).reshape(2, -1)
    batch = torch.arange(size).unsqueeze(1).expand(size, n_coords).flatten()
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = x0.clamp(0, width - 1)
    x1 = x1.clamp(0, width - 1)
    y0 = y0.clamp(0, height - 1)
    y1 = y1.clamp(0, height - 1)

    im_a = im[batch, :, y0, x0]
    im_b = im[batch, :, y1, x0]
    im_c = im[batch, :, y0, x1]
    im_d = im[batch, :, y1, x1]

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    im_new = wa * im_a + wb * im_b + wc * im_c + wd * im_d
    im_new = im_new.view(size, height, width, n_ch).permute((0, 3, 1, 2))
    return im_new


def warp(x, map_coords):
    size, _, height, width = x.shape
    x_coords = torch.arange(width, device=x.device).unsqueeze(
        0).expand(height, -1).flatten()
    y_coords = torch.arange(height, device=x.device).unsqueeze(
        1).expand(-1, width).flatten()
    n_coords = height * width
    coords = torch.stack((x_coords, y_coords)).unsqueeze(
        0).expand(size, 2, n_coords)
    coords_new = map_coords(coords).to(x.device)
    x_new = bilinear_interpolate(x, coords_new)
    return x_new


def swirl_full(x, hparams, same_on_batch=False):
    assert not same_on_batch
    size, _, height, width = x.shape
    device = x.device
    min_rad, max_rad = hparams['min_rad'], hparams['max_rad']
    min_strength, max_strength = hparams['min_strength'], hparams['max_strength']
    center = torch.stack((torch.randint(height, (size,)),
                          torch.randint(width, (size,)))).T.to(device).float()
    rad = torch.zeros(size, device=device).uniform_(min_rad, max_rad)
    strength = torch.zeros(size, device=device).uniform_(min_strength, max_strength)
    rotation = torch.zeros(size, device=device)
    x_new = warp(x, lambda coords: map_swirl(coords, center, rotation, strength, rad))
    params = torch.stack((
        normalize(center[:, 0], 0, height),
        normalize(center[:, 1], 0, width),
        normalize(rad, min_rad, max_rad),
        normalize(strength, min_strength, max_strength)),
        dim=1)
    return x_new, params


def swirl(x, hparams, same_on_batch=False):
    """Helper function that sets parameters based on alpha only and calls
    the full swirl function."""
    height = x.size(2)
    alpha = hparams['alpha']
    hparams['min_rad'] = 0
    hparams['max_rad'] = alpha * height * 2
    hparams['min_strength'] = 0
    hparams['max_strength'] = alpha * height
    return swirl_full(x, hparams, same_on_batch=same_on_batch)


def zoom_group(x, hparams, same_on_batch=False):
    size, n_ch, height, width = x.shape
    min_width, max_width = hparams['min_width'], hparams['max_width']
    min_height, max_height = hparams['min_height'], hparams['max_height']
    crop_width = torch.randint(min_width, max_width, (size, 1))
    crop_height = torch.randint(min_height, max_height, (size, 1))
    max_left = width - crop_width
    max_top = height - crop_height
    left = (torch.rand(size, 1) * max_left).round().int()
    top = (torch.rand(size, 1) * max_top).round().int()
    x_new = torch.zeros(x.shape)
    for i in range(size):
        t, l = top[i], left[i]
        cw, ch = crop_width[i], crop_height[i]
        x_new[i] = T.resized_crop(x[i], t, l, ch, cw, (height, width))
    params = torch.cat((
        normalize(top, 0, height - min_height),
        normalize(left, 0, width - min_width),
        normalize(crop_height, min_height, max_height),
        normalize(crop_width, min_width, max_width)
    ), dim=1)
    x_new = x_new.to(x.device)
    return x_new, params


blur_gaussian = batchify(blur_gaussian_single)
blur_mean = batchify(blur_mean_single)
blur_median = batchify(blur_median_single)
blur_mean_bilateral = batchify(blur_mean_bilateral_single)

# Vector version uses experimental torch.vmap feature
# blur_gaussian = batchify_vector(blur_gaussian_single)
# blur_mean = batchify_vector(blur_mean_single)
# blur_median = batchify_vector(blur_median_single)
# blur_mean_bilateral = batchify_vector(blur_mean_bilateral_single)


name_to_apply_tf = {
    'affine': transform_affine,
    'colorjitter': jitter_color,
    'gamma': alter_gamma,
    'sharp': alter_sharpness,
    'fft': fft,
    'precision': reduce_color_precision_diff,
    # Noise
    'drop_pixel': drop_pixel,
    'pepper': drop_pixel,   # alias to drop_pixel
    'salt': add_salt_noise,
    'normal': add_gaussian_noise,
    'poisson': add_poisson_noise,
    'uniform': add_uniform_noise,
    'speckle': add_speckle_noise,
    # Colorspace
    'xyz': alter_xyz,
    'yuv': alter_yuv,
    'lab': alter_lab,
    'hsv': alter_hsv,
    'gray': mix_gray_scale,
    'gray1': mix_one_partial_gray_scale,
    'gray2': mix_two_thirds_gray_scale,
    'graymix': mix_partial_gray_scale,
    # ================== Transforms that require initialization ============= #
    # Flipping
    'hflip': apply_module,
    'vflip': apply_module,
    # Blur filter
    # 'boxblur': blur_mean,
    # 'gaussblur': blur_gaussian,
    'boxblur': apply_module,
    'gaussblur': apply_module,
    # 'boxblur_same': apply_module,
    # 'gaussblur_same': apply_module,
    'medblur': apply_module,
    'motionblur': apply_module,
    # Edge detection
    'laplacian': apply_module_normalize,
    'sobel': apply_module_normalize,
    # Etc.
    'crop': apply_module,
    'edsr': apply_module,
    'erase': apply_module,
    'grayscale': apply_module,
    'solarize': apply_module,
    'jpeg': jpeg_compress,
    # ================================ Batchify ============================= #
    'gaussblur_batch': blur_gaussian,
    'medblur_batch': blur_median,
    'boxblur_batch': blur_mean,
    'bilatblur_batch': blur_mean_bilateral,
    'swirl': swirl,
    # ============================= Full version ============================ #
    'fft_full': fft_full,
    'precision_full': reduce_color_precision_diff_full,
    'jpeg_full': jpeg_compress_full,
    'xyz_color_full': alter_xyz_full,
    'yuv_color_full': alter_yuv_full,
    'lab_color_full': alter_lab_full,
    'hsv_color_full': alter_hsv_diff_full,
    'swirl_full': swirl_full,
    # ============================== Deprecated ============================= #
    # 'rotate': rotate,
    # 'translate': translate,
    # 'scale': scale,
    # 'shear': shear,
    # 'sp_noise': add_sp_noise,
    'zoom': zoom_group,
}

# =========================================================================== #
#                               Not Differentiable                            #
# =========================================================================== #


def equalize_histogram(x, hparams):
    x_new = kornia.enhance.equalize(x)
    x_new = x_new.to(x.device)
    return x_new, None


def reduce_color_precision(x, hparams, same_on_batch=False):
    size, n_ch, height, width = x.shape
    min_val, max_val = hparams['min'], hparams['max']
    scales = torch.randint(min_val, max_val + 1, (size, n_ch), device=x.device)
    scales_dim = scales.view(size, n_ch, 1, 1)
    x_new = x * scales_dim
    x_new = torch.round(x_new) / scales_dim
    params = normalize(scales, min_val, max_val)
    x_new = x_new.to(x.device)
    return x_new, params


def sketch_contrast_single(im, hparams):
    n_ch, height, width = im.shape
    im = to_numpy(im)
    low_perc_max, hi_perc_min = hparams['low_perc_max'], hparams['hi_perc_min']
    per_channel = np.random.choice(2) == 0
    params = [per_channel]
    low_percentile = np.random.uniform(0.01, low_perc_max, n_ch)
    hi_percentile = np.random.uniform(hi_perc_min, 0.99, n_ch)
    if per_channel:
        params += list(normalize(low_percentile, 0.01, low_perc_max)) + \
            list(normalize(hi_percentile, hi_perc_min, 0.99))
        im_new = np.zeros(im.shape)
        for i in range(n_ch):
            p2, p98 = np.percentile(
                im[i, :, :], (low_percentile[i] * 100, hi_percentile[i] * 100))
            im_new[i, :, :] = skimage.exposure.rescale_intensity(
                im[i, :, :], in_range=(p2, p98))
    else:
        params += [normalize(low_percentile[0], 0.01, low_perc_max)] * 3 + \
            [normalize(hi_percentile[0], hi_perc_min, 0.99)] * 3
        p2, p98 = np.percentile(
            im, (low_percentile[0] * 100, hi_percentile[0] * 100))
        im_new = skimage.exposure.rescale_intensity(im, in_range=(p2, p98))
    im_new = from_numpy(im_new)
    return im_new, params


def denoise_chambolle_single(im, hparams):
    im = to_numpy(im)
    min_weight, max_weight = hparams['min'], hparams['max']
    weight = np.random.uniform(min_weight, max_weight)
    multi_channel = np.random.choice(2) == 0
    im_new = denoise_tv_chambolle(
        im, weight=weight, multichannel=multi_channel)
    params = [normalize(weight, min_weight, max_weight), multi_channel]
    im_new = from_numpy(im_new)
    return im_new, params


def denoise_nonlocal_means_single(im, hparams):
    im = to_numpy(im)
    min_val, max_val = hparams['min'], hparams['max']

    h_1 = np.random.rand()
    params = [h_1]
    sigma_est = np.mean(
        skimage.restoration.estimate_sigma(im, multichannel=True))

    h = (max_val - min_val) * sigma_est * h_1 + min_val * sigma_est

    # If false, it assumes some weird 3D stuff
    multi_channel = np.random.choice(2) == 0
    params.append(multi_channel)

    # Takes too long to run without fast mode.
    fast_mode = True
    patch_size = np.random.randint(5, 7)
    params.append(patch_size)
    patch_distance = np.random.randint(6, 11)
    params.append(patch_distance)
    if multi_channel:
        im = skimage.restoration.denoise_nl_means(
            im, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode)
    else:
        for i in range(3):
            sigma_est = np.mean(skimage.restoration.estimate_sigma(
                im[i, :, :], multichannel=True))
            h = (max_val - min_val) * sigma_est * \
                params[i] + min_val * sigma_est
            im[i, :, :] = skimage.restoration.denoise_nl_means(
                im[i, :, :], h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode)

    im_new = from_numpy(im)
    return im_new, params


denoise_chambolle = batchify(denoise_chambolle_single)
sketch_contrast = batchify(sketch_contrast_single)
denoise_nonlocal_means = batchify(denoise_nonlocal_means_single)

name_to_nondiff_tf = {
    'equalize': apply_module,
    'contrast_sketch': sketch_contrast,
    'tv_denoise': denoise_chambolle,
    'nl_means_denoise': denoise_nonlocal_means_single,
    'reduce_cp': reduce_color_precision,
    # 'perturbe_fft': perturbe_fft,
    # 'jpeg_noise': add_jpeg_noise,
}

# =========================================================================== #
#                  Helper function for applying transforms                    #
# =========================================================================== #


def set_hparams(transform, hparams, module):
    if module is None:
        return lambda x, same_on_batch=False: transform(
            x, hparams, same_on_batch=same_on_batch)
    return lambda x, same_on_batch=False: transform(
        x, hparams, module, same_on_batch=same_on_batch)


def get_transform(tf, hparams, with_params=True, same_on_batch=False):
    assert tf in name_to_apply_tf, f'{tf} not implemented.'
    transform = name_to_apply_tf[tf]
    hparams = deepcopy(hparams)
    if 'alpha' in hparams:
        # Clip alpha to be slightly above zero
        hparams['alpha'] = max(hparams['alpha'], 0.001)
    if 'std' in hparams:
        hparams['std'] = max(hparams['std'], 0.001)
    module = None
    if tf in name_to_init_tf:
        module = init_transform(tf, hparams, same_on_batch=same_on_batch)
    transform = set_hparams(transform, hparams, module)
    if not with_params:
        return lambda x: transform(x, same_on_batch=same_on_batch)[0]
    return transform


def get_bpda_transform(bpda_params, device, same_on_batch=False):
    path = bpda_params['path']
    with open(os.path.join(path, 'transform.yml')) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    tf = config['transform']
    hparams = config[tf]
    module = None
    transform = get_transform(tf, hparams, same_on_batch=same_on_batch)

    if bpda_params.get('arc', None) != 'identity':
        bpda = get_bpda_network(hparams)
        bpda.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        bpda = bpda.to(device)

    class Transform(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x_new, params = transform(x)
            ctx.save_for_backward(x, params)
            return x_new

        @staticmethod
        @torch.enable_grad()
        def backward(ctx, grad_output):
            if bpda_params.get('arc', None) == 'identity':
                grad = grad_output
            else:
                x, params = ctx.saved_tensors
                x_pred = bpda(x, params)
                grad = torch.autograd.grad(x_pred, x, grad_output)
            return grad

    return lambda x: Transform.apply(x)

# =========================================================================== #
#                                   DEPRECATED                                #
# =========================================================================== #


def add_jpeg_noise_single(im, hparams):
    min_qual, max_qual = hparams['min'], hparams['max']
    quality = np.random.randint(min_qual, max_qual + 1)
    im = to_numpy(im)
    pil_image = PIL.Image.fromarray((im * 255).astype(np.uint8))
    f = BytesIO()
    pil_image.save(f, format='jpeg', quality=quality)
    im_new = np.asarray(PIL.Image.open(f)).astype(np.float32) / 255.0
    im_new = from_numpy(im_new)
    im_params = [normalize(quality, min_qual, max_qual)]
    return im_new, im_params


add_jpeg_noise = batchify(add_jpeg_noise_single)


def swirl_single_old(im, hparams):
    n_ch, height, width = im.shape
    min_rad, max_rad = hparams['min_rad'], hparams['max_rad']
    min_strength, max_strength = hparams['min_strength'], hparams['max_strength']
    center = (np.random.randint(0, height), np.random.randint(0, width))
    n_ch, height, width = im.shape
    im = to_numpy(im)
    rad = np.random.randint(min_rad, max_rad + 1)
    strength = np.random.randint(min_strength, max_strength + 1)
    im_new = skswirl(im, strength=strength, radius=rad, center=center)
    im_new = from_numpy(im_new)
    im_params = [normalize(rad, min_rad, max_rad),
                 normalize(strength, min_strength, max_strength),
                 normalize(center[0], 0, height - 1),
                 normalize(center[1], 0, width - 1)]
    return im_new, im_params


swirl_old = batchify(swirl_single_old)


def perturbe_fft_single(im, hparams):
    n_ch, height, width = im.shape
    im = to_numpy(im)
    min_pf, max_pf = hparams['min_pf'], hparams['max_pf']
    min_frac, max_frac = hparams['min_frac'], hparams['max_frac']
    # Everyone gets the same factor to avoid too many weird artifacts
    point_factor = np.random.uniform(min_pf, max_pf)
    randomized_mask = [np.random.choice(2) == 0 for _ in range(n_ch)]
    keep_fraction = np.random.uniform(min_frac, max_frac, n_ch)
    im_new = np.zeros(im.shape)
    for i in range(n_ch):
        im_fft = scipy.fft.fft2(im[:, :, i].reshape((height, width)))
        # Set r and c to be the number of rows and columns of the array.
        # r, c = im_fft.shape
        if randomized_mask[i]:
            mask = np.ones(im_fft.shape[:2]) > 0
            im_fft[int(height * keep_fraction[i]):int(height * (1 - keep_fraction[i]))] = 0
            im_fft[:, int(width * keep_fraction[i]):int(width * (1 - keep_fraction[i]))] = 0
            mask = ~mask
            # Now things to keep = 0, things to remove = 1
            mask = mask * \
                ~(np.random.uniform(size=im_fft.shape[:2]) < keep_fraction[i])
            # Now switch back
            mask = ~mask
            im_fft = np.multiply(im_fft, mask)
        else:
            im_fft[int(height * keep_fraction[i]):int(height * (1 - keep_fraction[i]))] = 0
            im_fft[:, int(width * keep_fraction[i]):int(width * (1 - keep_fraction[i]))] = 0
        # Now, lets perturb all the rest of the non-zero values by a relative factor
        im_fft = np.multiply(im_fft, point_factor)
        im_ch_new = scipy.fft.ifft2(im_fft).real
        # FFT inverse may no longer produce exact same range, so clip it back
        im_ch_new = np.clip(im_ch_new, 0, 1)

        im_new[:, :, i] = im_ch_new
    im_new = from_numpy(im_new)
    params = randomized_mask + list(normalize(keep_fraction, min_frac, max_frac)) + [
        normalize(point_factor, min_pf, max_pf)]
    return im_new, params


perturbe_fft = batchify(perturbe_fft_single)


def add_sp_noise(x, hparams, same_on_batch=False):
    # only works for images in 0-1 range
    size, n_ch, height, width = x.shape
    prob = hparams['prob']
    per_channel = np.random.rand() < 0.5
    if per_channel:
        salt = (torch.rand(size, n_ch, height, width)
                < prob / (2.0 * n_ch)).to(x.device)
        pepper = (torch.rand(size, n_ch, height, width)
                  < prob / (2.0 * n_ch)).to(x.device)
    else:
        salt = (torch.rand(size, 1, height, width) < prob / 2.0).to(x.device)
        pepper = (torch.rand(size, 1, height, width) < prob / 2.0).to(x.device)

    x_new = x * ~pepper * ~salt + salt
    x_new = x_new.to(x.device)
    return x_new, None


def rotate(x, hparams, same_on_batch=False):
    rotate = torch.eye(3, device=x.device).repeat(x.shape[0], 1, 1)
    angle = torch.zeros(x.shape[0], device=x.device)
    alpha = hparams['alpha'] / 180 * np.pi
    distr = hparams.get('dist', 'uniform')
    if distr == 'uniform':
        angle.uniform_(- alpha, alpha)
    elif distr == 'normal':
        angle.normal_(0, alpha)
    rotate[:, 0, 0] = torch.cos(angle)
    rotate[:, 1, 1] = torch.cos(angle)
    rotate[:, 0, 1] = torch.sin(angle)
    rotate[:, 1, 0] = -torch.sin(angle)
    grid = F.affine_grid(rotate[:, :2, :], x.size())
    x_new = F.grid_sample(x, grid)
    params = angle.unsqueeze(1)
    return x_new, params


def translate(x, hparams, same_on_batch=False):
    translate = torch.eye(3, device=x.device).repeat(x.shape[0], 1, 1)
    alpha = hparams['alpha']
    params = torch.zeros((x.shape[0], 2), device=x.device)
    distr = hparams.get('dist', 'uniform')
    if distr == 'uniform':
        params.uniform_(- alpha, alpha)
    elif distr == 'normal':
        params.normal_(0, alpha)
    translate[:, 0, 2] = params[:, 0]
    translate[:, 1, 2] = params[:, 1]
    grid = F.affine_grid(translate[:, :2, :], x.size())
    x_new = F.grid_sample(x, grid)
    return x_new, params


def scale(x, hparams, same_on_batch=False):
    scale = torch.eye(3, device=x.device).repeat(x.shape[0], 1, 1)
    alpha = hparams['alpha']
    params = torch.zeros((x.shape[0], 2), device=x.device)
    distr = hparams.get('dist', 'uniform')
    if distr == 'uniform':
        params.uniform_(1 - alpha, 1 + alpha)
    elif distr == 'normal':
        params.normal_(0, alpha)
    scale[:, 0, 0] = params[:, 0]
    scale[:, 1, 1] = params[:, 1]
    grid = F.affine_grid(scale[:, :2, :], x.size())
    x_new = F.grid_sample(x, grid)
    return x_new, params


def shear(x, hparams, same_on_batch=False):
    shear = torch.eye(3, device=x.device).repeat(x.shape[0], 1, 1)
    alpha = hparams['alpha']
    distr = hparams.get('dist', 'uniform')
    if distr == 'uniform':
        shear[:, 0, 1].uniform_(- alpha, alpha)
        shear[:, 1, 0].uniform_(- alpha, alpha)
    elif distr == 'normal':
        shear[:, 0, 1].normal_(0, alpha)
        shear[:, 1, 0].normal_(0, alpha)
    grid = F.affine_grid(shear[:, :2, :], x.size())
    x_new = F.grid_sample(x, grid)
    return x_new, None


def blur_mean_bilateral_single_old(im, hparams):
    n_ch, height, width = im.shape
    im.clamp_(0, 1)
    im = skimage.util.img_as_ubyte(to_numpy(im))
    min_rad, max_rad = hparams['min_rad'], hparams['max_rad']
    min_s, max_s = hparams['min_s'], hparams['max_s']

    radius = np.random.randint(min_rad, max_rad, size=n_ch)
    s0 = np.random.randint(min_s, max_s, size=n_ch)
    s1 = np.random.randint(min_s, max_s, size=n_ch)
    im_new = np.zeros(im.shape)
    for i in range(3):
        mask = disk(radius[i])
        im_new[:, :, i] = skimage.filters.rank.mean_bilateral(
            im[:, :, i], mask, s0=s0[i], s1=s1[i]) / 255.

    im_new = from_numpy(im_new)
    params = np.concatenate((normalize(radius, min_rad, max_rad),
                             normalize(s0, min_s, max_s),
                             normalize(s1, min_s, max_s)))
    return im_new, params


blur_mean_bilateral_old = batchify(blur_mean_bilateral_single_old)

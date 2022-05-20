'''
Code adapted from https://github.com/hendrycks/imagenet-r/tree/master/DeepAugment
'''

import os
import random
import sys
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as trnF
from torch.nn.functional import conv2d, gelu
from torchvision import datasets
from torchvision import transforms as trn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1,
                 bias=False, bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False,
                 act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, scale=4, res_scale=1,
                 rgb_range=255, n_colors=3, conv=default_conv):
        super(EDSR, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, pre_distortions={None}, body_distortions={None}):
        # print("Using FF pre distortions = ", pre_distortions)
        # print("Using FF body distortions = ", body_distortions)
        x = self.sub_mean(x)
        x = self.head(x)

        ######################################################################
        # PRE - DISTORTIONS
        ######################################################################

        # if 1 in pre_distortions:
        #     for _ in range(5):
        #         c1, c2 = random.randint(0, 63), random.randint(0, 63)
        #         x[:, c1], x[:, c2] = x[:, c2], x[:, c1]

        # if 2 in pre_distortions:
        #     # Random matrix of 1s and 0s
        #     rand_filter_weight = torch.round(torch.rand_like(x) + 0.45)
        #     x = x * rand_filter_weight

        # if 3 in pre_distortions:
        #     # Random matrix of 1s and -1s
        #     rand_filter_weight = (torch.round(
        #         torch.rand_like(x) + 0.475) * 2) - 1
        #     x = x * rand_filter_weight

        const = 0.05
        mean = x.mean().detach()
        scale = ((x - mean) ** 2).mean().detach()
        x += torch.randn_like(x) * const * scale

        ######################################################################
        # BODY - DISTORTIONS
        ######################################################################

        # if 1 in body_distortions:
        #     res = self.body[:5](x)
        #     res = -res
        #     res = self.body[5:](res)
        # elif 2 in body_distortions:
        #     if random.randint(0, 2) == 1:
        #         act = F.relu
        #     else:
        #         act = F.gelu
        #     res = self.body[:5](x)
        #     res = act(res)
        #     res = self.body[5:](res)
        # elif 3 in body_distortions:
        #     if random.randint(0, 2) == 1:
        #         axes = [1, 2]
        #     else:
        #         axes = [1, 3]
        #     res = self.body[:5](x)
        #     res = torch.flip(res, axes)
        #     res = self.body[5:](res)
        # elif 4 in body_distortions:
        #     to_skip = set([random.randint(2, 16) for _ in range(3)])
        #     for i in range(len(self.body)):
        #         if i not in to_skip:
        #             res = self.body[i](x)
        # else:
        #     res = self.body(x)

        res = self.body(x)
        mean = res.mean().detach()
        scale = ((res - mean) ** 2).mean().detach()
        res += torch.randn_like(res) * const * scale

        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x


def get_weights():
    weights = torch.load(
        '/home/chawin/rand-smooth/saved_models/edsr_baseline_x4.pt')

    random_sample_list = np.random.randint(0, 17, size=3)
    for option in list(random_sample_list):
        if option == 0:
            i = np.random.choice(np.arange(0, 10, 3))
            weights['body.' + str(i) + '.body.0.weight'] = torch.flip(
                weights['body.' + str(i) + '.body.0.weight'], (0,))
            weights['body.' + str(i) + '.body.0.bias'] = torch.flip(
                weights['body.' + str(i) + '.body.0.bias'], (0,))
            weights['body.' + str(i) + '.body.2.weight'] = torch.flip(
                weights['body.' + str(i) + '.body.2.weight'], (0,))
            weights['body.' + str(i) + '.body.2.bias'] = torch.flip(
                weights['body.' + str(i) + '.body.2.bias'], (0,))
        elif option == 1:
            i = np.random.choice(np.arange(1, 10, 3))
            weights['body.' + str(i) + '.body.0.weight'] = - \
                weights['body.' + str(i) + '.body.0.weight']
            weights['body.' + str(i) + '.body.0.bias'] = - \
                weights['body.' + str(i) + '.body.0.bias']
        elif option == 2:
            i = np.random.choice(np.arange(0, 10, 3))
            weights['body.' + str(i) + '.body.0.weight'] = 0 * \
                weights['body.' + str(i) + '.body.0.weight']
            weights['body.' + str(i) + '.body.0.bias'] = 0 * \
                weights['body.' + str(i) + '.body.0.bias']
        elif option == 3:
            i = np.random.choice(np.arange(0, 10, 3))
            weights['body.' + str(i) + '.body.0.weight'] = - \
                gelu(weights['body.' + str(i) + '.body.0.weight'])
            weights['body.' + str(i) + '.body.2.weight'] = - \
                gelu(weights['body.' + str(i) + '.body.2.weight'])
        elif option == 4:
            i = np.random.choice(np.arange(0, 10, 3))
            weights['body.' + str(i) + '.body.0.weight'] = weights['body.' + str(i) + '.body.0.weight'] * \
                torch.Tensor([[0, 1, 0], [1, -4., 1], [0, 1, 0]]
                             ).view(1, 1, 3, 3).cuda()
        elif option == 5:
            i = np.random.choice(np.arange(0, 10, 3))
            weights['body.' + str(i) + '.body.0.weight'] = weights['body.' + str(i) + '.body.0.weight'] * \
                torch.Tensor([[-1, -1, -1], [-1, 8., -1], [-1, -1, -1]]
                             ).view(1, 1, 3, 3).cuda()
        elif option == 6:
            i = np.random.choice(np.arange(0, 10, 3))
            weights['body.' + str(i) + '.body.2.weight'] = weights['body.' + str(i) + '.body.2.weight'] * \
                (1 + 2 * np.float32(np.random.uniform()) *
                 (2 * torch.rand_like(weights['body.' + str(i) + '.body.2.weight'] - 1)))
        elif option == 7:
            i = np.random.choice(np.arange(0, 10, 3))
            weights['body.' + str(i) + '.body.0.weight'] = torch.flip(
                weights['body.' + str(i) + '.body.0.weight'], (-1,))
            weights['body.' + str(i) + '.body.2.weight'] = - \
                1 * weights['body.' + str(i) + '.body.2.weight']
        elif option == 8:
            i = np.random.choice(np.arange(1, 13, 4))
            z = torch.zeros_like(weights['body.' + str(i) + '.body.0.weight'])
            for j in range(z.size(0)):
                shift_x, shift_y = np.random.randint(3, size=(2,))
                z[:, j, shift_x, shift_y] = np.random.choice([1., -1.])
            weights['body.' + str(i) + '.body.0.weight'] = conv2d(
                weights['body.' + str(i) + '.body.0.weight'], z, padding=1)
        elif option == 9:
            i = np.random.choice(np.arange(0, 10, 3))
            z = (2 * torch.rand_like(weights['body.' + str(i) + '.body.0.weight'])
                 * np.float32(np.random.uniform()) - 1) / 6.
            weights['body.' + str(i) + '.body.0.weight'] = conv2d(
                weights['body.' + str(i) + '.body.0.weight'], z, padding=1)
        elif option == 10:
            i = np.random.choice(np.arange(1, 12, 4))
            z = torch.FloatTensor(np.random.dirichlet(
                [0.1] * 9, (64, 64))).view(64, 64, 3, 3).cuda()  # 2.weight
            weights['body.' + str(i) + '.body.2.weight'] = conv2d(
                weights['body.' + str(i) + '.body.2.weight'], z, padding=1)
        elif option == 11:
            i = random.choice(list(range(15)))
            noise = (torch.rand_like(
                weights['body.' + str(i) + '.body.2.weight']) - 0.5) * 1.0
            weights['body.' + str(i) + '.body.2.weight'] += noise
        elif option == 12:
            _ij = [[random.choice(list(range(15))), random.choice([0, 2])]
                   for _ in range(5)]
            for i, j in _ij:
                _k = random.randint(1, 3)
                if random.randint(0, 1) == 0:
                    _dims = (2, 3)
                else:
                    _dims = (0, 1)
                weights['body.' + str(i) + '.body.' + str(j) + '.weight'] = torch.rot90(
                    weights['body.' + str(i) + '.body.' + str(j) + '.weight'], k=_k, dims=_dims)
        elif option == 13:
            _i = [random.choice(list(range(15))) for _ in range(5)]
            for i in _i:
                rand_filter_weight = torch.round(torch.rand_like(
                    weights['body.' + str(i) + '.body.0.weight'])) * 2 - 1  # Random matrix of 1s and -1s
                weights['body.' + str(i) + '.body.0.weight'] = weights['body.' +
                                                                       str(i) + '.body.0.weight'] * rand_filter_weight
        elif option == 14:
            # Barely noticable difference here
            _i = [random.choice(list(range(15))) for _ in range(5)]
            for i in _i:
                rand_filter_weight = torch.round(torch.rand_like(
                    weights['body.' + str(i) + '.body.0.weight']))  # Random matrix of 1s and 0s
                weights['body.' + str(i) + '.body.0.weight'] = weights['body.' +
                                                                       str(i) + '.body.0.weight'] * rand_filter_weight
        elif option == 15:
            # Negate some entire filters. Definitely a noticable difference
            _i = [random.choice(list(range(15))) for _ in range(5)]
            for i in _i:
                filters_to_be_zeroed = [random.choice(
                    list(range(64))) for _ in range(32)]
                weights['body.' +
                        str(i) + '.body.0.weight'][filters_to_be_zeroed] *= -1
        elif option == 16:
            # Only keep the max filter value in the conv
            _ij = [[random.choice(list(range(15))), random.choice([0, 2])]
                   for _ in range(5)]
            for i, j in _ij:
                w = torch.reshape(
                    weights['body.' + str(i) + '.body.' + str(j) + '.weight'], shape=(64, 64, 9))
                res = torch.topk(w, k=1)

                w_new = torch.zeros_like(w).scatter(2, res.indices, res.values)
                w_new = w_new.reshape(64, 64, 3, 3)
                weights['body.' + str(i) + '.body.' +
                        str(j) + '.weight'] = w_new
        else:
            raise NotImplementedError()
    return weights

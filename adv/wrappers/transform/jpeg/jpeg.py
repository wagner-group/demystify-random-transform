'''
This code implements differentiable JPEG compression and is modified from
https://github.com/mlomnitz/DiffJPEG/blob/master/DiffJPEG.py.
Now `quality` can vary between samples in the same batch.
'''
import torch
import torch.nn as nn

from .compression import compress_jpeg
from .decompression import decompress_jpeg
from .utils import diff_round, quality_to_factor


class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        # TODO: Both compress and decompress use a very poor fix on the
        # DataParallel issue: nn.Parameter (or buffer) is not copied to all
        # devices properly.
        self.compress = compress_jpeg(rounding=rounding)
        self.decompress = decompress_jpeg(height, width, rounding=rounding)

    def forward(self, x, quality=80):
        if isinstance(quality, int):
            assert 0 <= quality <= 100
        elif isinstance(quality, torch.Tensor):
            assert quality.ndim == 1 and quality.size(0) == x.size(0)
        else:
            raise ValueError('quality is invalid!')

        factor = quality_to_factor(quality)
        y, cb, cr = self.compress(x, factor=factor)
        recovered = self.decompress(y, cb, cr, factor=factor)
        return recovered

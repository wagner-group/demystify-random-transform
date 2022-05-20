import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, normalize=None):
        super(Normalize, self).__init__()
        self.normalize = normalize
        if normalize is not None:
            mean = torch.tensor(normalize['mean'])[(..., ) + (None, None)]
            std = torch.tensor(normalize['std'])[(..., ) + (None, None)]
            self.register_buffer('norm_mean', mean)
            self.register_buffer('norm_std', std)

    def forward(self, x):
        if self.normalize is None:
            return x
        return (x - self.norm_mean) / self.norm_std

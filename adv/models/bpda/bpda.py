import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, skip=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.skip = skip

    def forward(self, z, x):
        out = F.relu(self.bn1(self.conv1(z)))
        if self.skip:
            out = torch.cat((x, out), dim=1)
        return out


class BPDA(nn.Module):
    def __init__(self, n_params):
        super(BPDA, self).__init__()
        n_ch = n_params + 5
        self.n_params = n_params

        self.db1 = DenseBlock(n_ch, 16)
        self.db2 = DenseBlock(16 + n_ch, 22)
        self.db3 = DenseBlock(22 + n_ch, 32)
        self.db4 = DenseBlock(32 + n_ch, 45)
        self.db5 = DenseBlock(45 + n_ch, 64)
        self.db6 = DenseBlock(64 + n_ch, 3, kernel_size=3, skip=False)

    def forward(self, x, params):
        size, n_ch, height, width = x.shape
        if params is not None:
            params_full = params.view(size, self.n_params, 1, 1) \
                                .expand(size, self.n_params, height, width)
        else:
            params_full = torch.FloatTensor(size, 0, height, width)
        params_full = params_full.to(x.device)

        row_coords = torch.arange(-1, 1, 2 / height) \
                          .view(height, 1) \
                          .expand(size, 1, height, width) \
                          .to(x.device)
        col_coords = torch.arange(-1, 1, 2 / width) \
                          .expand(size, 1, height, width) \
                          .to(x.device)

        inputs = torch.cat((x, row_coords, col_coords, params_full), dim=1)

        z = self.db1(inputs, inputs)
        z = self.db2(z, inputs)
        z = self.db3(z, inputs)
        z = self.db4(z, inputs)
        z = self.db5(z, inputs)
        z = self.db6(z, inputs)
        x_pred = torch.clamp(z, 0, 1)
        return x_pred


class LargeBPDA(nn.Module):
    def __init__(self, n_params, layers=40, n_kernels=8, kernel_size=7):
        super(LargeBPDA, self).__init__()
        n_ch = n_params + 5
        self.n_params = n_params

        self.blocks = nn.ModuleList()
        block = DenseBlock(n_ch, n_kernels, kernel_size=kernel_size)
        self.blocks.append(block)
        for i in range(layers):
            block = DenseBlock(n_kernels + n_ch, n_kernels,
                               kernel_size=kernel_size)
            self.blocks.append(block)
        block = DenseBlock(n_kernels + n_ch, 3,
                           kernel_size=kernel_size, skip=False)
        self.blocks.append(block)

    def forward(self, x, params):
        size, n_ch, height, width = x.shape
        if params is not None:
            params_full = params.view(size, self.n_params, 1, 1) \
                                .expand(size, self.n_params, height, width)
        else:
            params_full = torch.FloatTensor(size, 0, height, width)
        params_full = params_full.to(x.device)

        row_coords = torch.arange(-1, 1, 2 / height) \
                          .view(height, 1) \
                          .expand(size, 1, height, width) \
                          .to(x.device)
        col_coords = torch.arange(-1, 1, 2 / width) \
                          .expand(size, 1, height, width) \
                          .to(x.device)

        inputs = torch.cat((x, row_coords, col_coords, params_full), dim=1)

        z = inputs
        for block in self.blocks:
            z = block(z, inputs)
        x_pred = torch.clamp(z, 0, 1)
        return x_pred


class Identity(nn.Module):
    def forward(self, x, params):
        return x


def parse_bpda_name(config):
    tokens = [config['meta']['dataset'], 'bpda']
    if config['meta']['model_name'] is not None:
        tokens.append(config['meta']['model_name'])
    tf_name = config['transform']
    hparams = config[tf_name]
    tokens.append(tf_name)
    for hparam_name in hparams:
        tokens.append(hparam_name)
        tokens.append(str(hparams[hparam_name]))
    tokens.extend(('exp', str(config['meta']['exp_id'])))
    return '_'.join(tokens)


def get_bpda_network(hparams):
    arc = hparams['arc']
    nparams = hparams['nparams']
    if arc == 'large':
        net = LargeBPDA(nparams)
    else:
        net = BPDA(nparams)
    return net

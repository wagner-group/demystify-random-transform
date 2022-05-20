from .wrapper import Wrapper
from torch.nn import DataParallel


class BaseWrapper(Wrapper):
    """Base wrappper. Also applies torch.nn.DataParallel if given `device` is
    'cuda'.
    """

    def __init__(self, base_net, device):
        super(BaseWrapper, self).__init__()
        self.base_net = base_net
        if device == 'cuda':
            self.net = DataParallel(base_net)
        elif 'cuda' in device:
            self.net = DataParallel(base_net, device_ids=[device])
        else:
            self.net = base_net

    def forward(self, inputs, **kwargs):
        return self.net(inputs, **kwargs)

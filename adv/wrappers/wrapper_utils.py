'''General functionality related to wrapper models.'''
import torch
from torch.nn import DataParallel

from .rand_spheres_wrapper import RandSpheresWrapper
from .rand_wrapper import RandWrapper


def create_wrapper(base_net, config, mode, device=None):
    """Wrap Pytorch Module."""
    assert mode in ['train', 'test']
    if config['meta']['dataset'] == 'mnist':
        input_size = 28
    elif 'cifar' in config['meta']['dataset']:
        input_size = 32
    elif 'imagenet' in config['meta']['dataset']:
        input_size = 224
    else:
        raise NotImplementedError('Dataset not implemented.')

    # Apply RandWrapper should random transformations be used
    net = base_net
    if 'rand' in config['meta']['method']:
        if config['meta']['dataset'] == 'spheres':
            net = RandSpheresWrapper(base_net, config['rand'], input_size)
        else:
            net = RandWrapper(base_net, config['rand'], input_size, device=device)

    # DataParallel is always applied
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        net = DataParallel(net.to('cuda'))
    elif 'cuda' in device:
        net = DataParallel(net, device_ids=[device])

    return net

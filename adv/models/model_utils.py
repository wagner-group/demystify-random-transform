'''General functionality related to Pytorch models.'''
import torch.nn as nn

from .mnist_model import BasicModel
from .normalize import Normalize
from .preact_resnet import PreActBlock, PreActResNet
from .resnet import resnet18, resnet34, resnet50
from .simple_model import DenseModel, DenseModelV2
from .wideresnet import WideResNet


def create_model(config, num_classes):
    """Build neural network."""
    params = config['meta']
    if params['network'] == 'basic':
        net = BasicModel(num_classes=num_classes)
    elif params['network'] == 'resnet':
        # Use PreActResNet-20
        net = PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif params['network'] == 'wideresnet':
        # Use WideResNet-34-10
        net = WideResNet(depth=34, num_classes=num_classes, widen_factor=10, dropRate=0.3)
    elif params['network'] in ('resnet18', 'resnet34', 'resnet50'):
        if params['network'] == 'resnet18':
            net = resnet18(pretrained=params['pretrained'], progress=False)
            linear_size = 512
        elif params['network'] == 'resnet34':
            net = resnet34(pretrained=params['pretrained'], progress=False)
            linear_size = 512
        else:
            net = resnet50(pretrained=params['pretrained'], progress=False)
            linear_size = 2048
        # Replace last layer if only retrain on subset of classes
        if params['classes'] is not None or params['dataset'] == 'imagenette':
            net.fc = nn.Linear(linear_size, num_classes)
    elif params['network'] == 'dense':
        net = DenseModel(params['d'], num_classes=num_classes)
    elif params['network'] == 'dense2':
        net = DenseModelV2(params['d'], num_classes=num_classes)
    else:
        raise NotImplementedError('Specified network not implemented.')

    return net


def add_normalization(net, params):
    if params['normalize'] is not None:
        norm = Normalize(params['normalize'])
        net = nn.Sequential(norm, net)
    return net

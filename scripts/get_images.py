'''Train MNIST model with adversarial training'''
from __future__ import print_function

import argparse
import os
from copy import deepcopy

import numpy as np
import torch.backends.cudnn as cudnn
import yaml
from torchvision.utils import save_image

from adv.models import add_normalization, create_model
from adv.utils import load_dataset, set_random_seed, setup_routine
from adv.wrappers import create_wrapper


def main(config_file):
    """Main function. Use config file train_and_test_DATASET.yml"""

    # Set CUDNN param
    cudnn.benchmark = True

    # Parse config file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['meta']['gpu_id']
    config['rand']['use_saved_transforms'] = False
    config, device, save_dir, log = setup_routine(
        config, 'get_images', load_config=True)

    # Set all random seeds
    set_random_seed(config['meta']['seed'])

    # Load dataset
    log.info('Preparing data...')
    (_, _, testloader), num_classes = load_dataset(config, dataloader=True)

    # Build neural network
    log.info('Building model...')
    basic_net = create_model(config, num_classes)
    basic_net = add_normalization(basic_net, config['meta']).to(device)
    # Wrap the neural network with module for random transformations or
    # adversarial training
    net = create_wrapper(basic_net, config, 'test')

    model_path = os.path.join(save_dir, 'model.pt')
    net.load_weights(model_path)
    net.to(device).eval()

    params = deepcopy(config['rand'])
    np.random.shuffle(params['transforms'])
    # params['transforms'] = ['crop']

    inputs = iter(testloader).next()[0].to(device)
    x = net.base_net.apply_transforms(inputs, params)
    for i in range(10):
        save_image(x[i], f'./images/tf{i}.png')
        save_image(inputs[i], f'./images/input{i}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test a model.')
    parser.add_argument(
        'config_file', type=str, help='name of config file')
    args = parser.parse_args()
    main(args.config_file)

'''Test model'''
from __future__ import print_function

import argparse
import datetime
import os
import pickle

import torch.backends.cudnn as cudnn
import yaml

from adv.models import add_normalization, create_model
from adv.utils import (load_dataset, save_outputs, set_random_seed,
                       setup_routine)
from adv.utils.diversity import compute_diversity
from adv.wrappers import create_wrapper


def main(config_file):
    """Main function. Use a specified config file through command line."""
    # Set CUDNN param
    cudnn.benchmark = True
    # cudnn.benchmark = False
    # cudnn.deterministic = True

    # Parse config file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['meta']['gpu_id']

    if 'rand' in config and config['rand'].get('same_on_batch', False):
        for mode in ('test', 'attack'):
            config['rand'][mode]['num_draws'] = 1
            config['rand'][mode]['rule'] = 'none'
            config['rand'][mode]['tf_order'] = 'fixed'
            config['rand'][mode]['fix_seed'] = True

    # Initialize save directory and logging
    ts = datetime.datetime.now().strftime('%m%d%y-%H%M%S')
    log_name = f"test_{config['meta']['test'].get('save_name', '')}_{ts}"
    config, device, save_dir, log = setup_routine(config, log_name,
                                                  load_config=True)

    # Set all random seeds
    set_random_seed(config['meta']['seed'])

    # Load dataset
    log.info('Preparing data...')
    (_, _, testloader), num_classes = load_dataset(config, dataloader=True)

    # Build neural network
    log.info('Building model...')
    basic_net = create_model(config, num_classes)
    basic_net = add_normalization(basic_net, config['meta'])
    # Wrap the neural network with module for random transformations or
    # adversarial training
    net = create_wrapper(basic_net, config, 'test')

    epoch = config['meta'].get('load_epoch', None)
    if epoch is not None:
        model_path = os.path.join(save_dir, f'model_epoch{epoch}.pt')
    else:
        model_path = os.path.join(save_dir, 'model.pt')
    if not os.path.exists(model_path):
        log.info('Model does not exist. Weights are randomly initialized.')
    elif not config.get('rand', {}).get('save_transformed_img', False):
        log.info(f'Loading model from {model_path}')
        net.module.load_weights(model_path)
    net.to(device).eval()

    log.info('Computing diversity metric(s)...')
    if config['diversity'].get('data_name', None) is not None:
        data_path = os.path.join(save_dir, config['diversity']['data_name'])
        saved_file = pickle.load(open(data_path, 'rb'))
        x_adv = saved_file['adv_0']['x_adv'][0]
    else:
        x_adv = None

    # Computing diversity on test data
    div = compute_diversity(config, net, testloader, x_adv=x_adv)
    print(div)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate accuracy and robustness of a model.')
    parser.add_argument(
        'config_file', type=str, help='name of config file')
    args = parser.parse_args()
    main(args.config_file)

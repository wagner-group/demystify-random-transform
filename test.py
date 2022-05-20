'''Test model'''
from __future__ import print_function

import argparse
import datetime
import os

import torch.backends.cudnn as cudnn
import yaml

from adv.models import add_normalization, create_model
from adv.utils import (load_dataset, save_outputs, set_random_seed,
                       setup_routine)
from adv.utils.test_utils import main_test
from adv.wrappers import create_wrapper


def main(config_file):
    """Main function. Use a specified config file through command line."""
    # Set CUDNN param. Use the commented options for deterministic runs
    cudnn.benchmark = True
    # cudnn.benchmark = False
    # cudnn.deterministic = True

    # Parse config file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    if 'rand' in config and config['rand'].get('same_on_batch', False):
        for mode in ('test', 'attack'):
            config['rand'][mode]['num_draws'] = 1
            config['rand'][mode]['rule'] = 'none'
            config['rand'][mode]['tf_order'] = 'fixed'
            config['rand'][mode]['fix_seed'] = True

    # Initialize save directory and logging
    ts = datetime.datetime.now().strftime('%m%d%y-%H%M%S')
    log_name = f"test_{config['meta']['test'].get('save_name', '')}_{ts}"
    config, device, save_dir, log = setup_routine(config, log_name, load_config=True)

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
    net = create_wrapper(basic_net, config, 'test', device)

    if config['meta'].get('simple_path', False):
        save_dir = os.path.join(config['meta']['save_path'], config['meta']['model_name'])
    epoch = config['meta'].get('load_epoch', None)
    if epoch is not None:
        model_path = os.path.join(save_dir, f'model_epoch{epoch}.pt')
    else:
        model_path = os.path.join(save_dir, 'model.pt')
    log.info(f'Loading model from "{save_dir}" ...')
    if not os.path.exists(model_path):
        log.info('Model does not exist. Weights are randomly initialized.')
    elif not config.get('rand', {}).get('save_transformed_img', False):
        net.module.load_weights(model_path)
    net.to(device).eval()

    # Call main test function
    return_output = (config['meta']['test']['save_clean_out'] or
                     config['meta']['test']['save_adv_out'])
    outputs = main_test(config, net, testloader, 'test', log,
                        return_adv=config['meta']['test']['save_adv'],
                        return_output=return_output,
                        clean_only=config['meta']['test']['clean_only'],
                        adv_only=config['meta']['test']['adv_only'])

    # Save specified outputs
    if (config['meta']['test']['save_output'] or return_output or
            config['meta']['test']['save_adv']):
        name = config['meta']['test'].get('save_name', ts)
        save_outputs(config, outputs, name=name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate accuracy and robustness.')
    parser.add_argument('config_file', type=str, help='name of config file')
    args = parser.parse_args()
    main(args.config_file)

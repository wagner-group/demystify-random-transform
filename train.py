'''Train MNIST model with adversarial training'''
from __future__ import print_function

import argparse
import os

import torch.backends.cudnn as cudnn
import yaml

from adv.utils import save_outputs
from adv.utils.test_utils import main_test
from adv.utils.train_utils import main_train
from adv.wrappers import RandWrapper


def main(config_file):
    """Main function. Use config file train_and_test_DATASET.yml"""

    # Set CUDNN param
    cudnn.benchmark = True

    # Parse config file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    if config['rand'].get('same_on_batch', False):
        for mode in ('train', 'test', 'attack'):
            config['rand'][mode]['num_draws'] = 1
            config['rand'][mode]['rule'] = 'none'
            config['rand'][mode]['tf_order'] = 'fixed'
            config['rand'][mode]['fix_seed'] = True

    # Call main train function
    net, config, (_, _, testloader), log = main_train(config, load_config=True)

    if isinstance(net.module, RandWrapper):
        config['rand']['use_saved_transforms'] = True
        if not config['rand'].get('same_on_batch', False):
            net.module.params['test']['num_draws'] = 20
            net.module.params['test']['tf_order'] = 'random'
            config['meta']['test']['num_conf_repeats'] = 10
    return_output = config['meta']['test']['save_clean_out'] or \
        config['meta']['test']['save_adv_out']

    # Call main test function
    outputs = main_test(config, net, testloader, 'test', log,
                        return_adv=config['meta']['test']['save_adv'],
                        return_output=return_output,
                        clean_only=config['meta']['test']['clean_only'])

    # Save specified outputs
    save_outputs(config, outputs)
    log.info('Finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('config_file', type=str, help='name of config file')
    args = parser.parse_args()
    main(args.config_file)

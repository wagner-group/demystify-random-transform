'''Train MNIST model with adversarial training'''
from __future__ import print_function

import argparse
import os
import pickle
import warnings
from copy import deepcopy

import ray
import torch.backends.cudnn as cudnn
import yaml
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.suggest.hyperopt import HyperOptSearch

from adv.utils import save_outputs, set_random_seed, setup_routine
from adv.utils.ray_utils import (CustomStopper, ray_report, ray_update_config,
                                 trial_name_string)
from adv.utils.test_utils import main_test
from adv.utils.train_utils import main_train

warnings.filterwarnings('ignore')


def trainable(config, main_config=None, checkpoint_dir=None):
    """Main train function called by Tune."""

    # TODO: Clean this up
    # Look up past trials
    # points = pickle.load(open(
    #     '/home/chawin/rand-smooth/save/cifar_rand10_ray_init.pkl', 'rb'))
    # for point in points:
    #     match = 0
    #     for tf in point['config']:
    #         match += abs(point['config'][tf] - config[tf]) <= 1e-3
    #     if match == len(config):
    #         tune.report(clean_acc=0, adv_acc=point['metric'],
    #                     weight_acc=point['metric'])
    #         return

    # Update config with tune search space
    config = ray_update_config(config, main_config)

    # Call main train function
    net, config, (_, validloader, _), log = main_train(config)

    # Call main test function
    return_output = config['meta']['test']['save_clean_out'] or \
        config['meta']['test']['save_adv_out']
    # TODO: Move to somewhere more explicit
    if not config['rand'].get('same_on_batch', False):
        net.module.params['test']['num_draws'] = 20
        net.module.params['test']['tf_order'] = 'random'
    outputs = main_test(config, net, validloader, 'ray', log,
                        return_adv=config['meta']['test']['save_adv'],
                        return_output=return_output)
    clean_val_acc = outputs['clean']['acc']
    adv_val_acc = outputs['adv']['acc']

    # Compute and report metrics
    ray_report(config, clean_val_acc, adv_val_acc)

    # Delete saved models and config before exiting
    save_dir = ray.tune.get_trial_dir()
    try:
        os.remove(os.path.join(save_dir, 'model.pt'))
    except OSError as e:
        log.error(f'Error: {save_dir} : {e.strerror}')


def main(config_file):

    # Parse config file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = config['ray']['gpu_id']

    # Set CUDNN param outside of main training function
    cudnn.benchmark = True

    # Overwrite some meta config with ray config
    config['meta']['exp_id'] = config['ray']['exp_id']
    config['meta']['train']['save_epochs'] = \
        config['ray']['metric']['report_epochs']

    # Initialize save directory and logging
    config, _, save_dir, log = setup_routine(config, 'ray', load_config=False)
    set_random_seed(config['meta']['seed'])

    # Specify Tune objective
    metric = {'metric': config['ray']['metric']['metric'], 'mode': 'max'}

    # Build search space
    search_space = {}
    point = {}
    for key in config['ray']['search_space']:
        val = config['ray']['search_space'][key]
        if len(val) == 2:
            search_space[key] = tune.uniform(*val)
        elif len(val) == 3:
            search_space[key] = tune.quniform(*val)
        else:
            raise ValueError(f'Invalid search space for {key}!')
        point[key] = 0.

    # Initial points to evaluate by the algorithm
    points_to_eval = None
    if config['ray']['init_eval_points']:
        num_points = config['ray']['init_eval_points']
        points_to_eval = []
        for i in range(num_points):
            new_point = deepcopy(point)
            for key in config['ray']['search_space']:
                val = config['ray']['search_space'][key]
                assert len(val) == 2
                new_point[key] = val[0] + (val[1] - val[0]) * i / num_points
            points_to_eval.append(new_point)
        print(points_to_eval)
        # points = pickle.load(open('save/cifar_rand10_ray_init.pkl', 'rb'))
        # points_to_eval = []
        # for point in points:
        #     points_to_eval.append(point['config'])

    # Set Tune scheduler
    scheduler = None
    if config['ray']['scheduler'] == 'asha':
        if config['ray']['asha']['max_t'] is None:
            config['ray']['asha']['max_t'] = config['meta']['train']['epochs']
        scheduler = ASHAScheduler(**metric, **config['ray']['asha'])
    elif config['ray']['scheduler'] == 'bohb':
        if config['ray']['bohb']['max_t'] is None:
            config['ray']['bohb']['max_t'] = config['meta']['train']['epochs']
        scheduler = HyperBandForBOHB(**metric, **config['ray']['bohb'])

    # Set Tune search algorithm
    algo = None
    if config['ray']['algo'] == 'bayes':
        algo = BayesOptSearch(random_state=config['meta']['seed'],
                              points_to_evaluate=points_to_eval, **metric,
                              **config['ray']['bayes'])
    elif config['ray']['algo'] == 'hyperopt':
        algo = HyperOptSearch(random_state_seed=config['meta']['seed'],
                              points_to_evaluate=points_to_eval,
                              **metric, **config['ray']['hyperopt'])
    elif config['ray']['algo'] == 'bohb':
        algo = TuneBOHB(points_to_evaluate=points_to_eval, **metric)
    elif config['ray']['algo'] == 'dragonfly':
        algo = DragonflySearch(optimizer='bandit', domain='euclidean',
                               points_to_evaluate=points_to_eval, **metric)
    elif config['ray']['algo'] == 'ax':
        algo = AxSearch(points_to_evaluate=points_to_eval, **metric)
    if algo is not None:
        max_concurrent = config['ray']['max_concurrent']
        algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent)

    # Set stopper for the entire experiment
    stopper = CustomStopper(**metric, **config['ray']['stopper'])

    log.info('Start running tune...')
    result = tune.run(
        tune.with_parameters(trainable, main_config=config),
        config=search_space,
        scheduler=scheduler,
        search_alg=algo,
        name='tune',
        trial_name_creator=trial_name_string,
        local_dir=save_dir,
        stop=stopper,
        **config['ray']['run_params'],
    )

    best_trial = result.get_best_trial(scope='all', **metric)
    log.info(f'Best trial config: {best_trial.config}')
    log.info(f'Best trial val accuracy (clean/adv): '
             f'{best_trial.last_result["clean_acc"]:.2f}/'
             f'{best_trial.last_result["adv_acc"]:.2f}.')

    # Update config with best hyperparameters
    config = ray_update_config(best_trial.config, config)
    pickle.dump(config['rand'], open(f'{save_dir}/rand.cfg', 'wb'))

    # ======================================================================= #
    #                Train a full model using the best config                 #
    # ======================================================================= #
    # TODO: move to config file?
    config['meta']['val_size'] = 0.1
    config['meta']['train']['batch_size'] = 128
    config['meta']['train']['epochs'] = 100
    config['meta']['train']['step_len'] = 10
    config['meta']['train']['eval_with_atk'] = False
    config['meta']['valid']['num_samples'] = float('inf')
    config['rand']['train']['num_draws'] = 4
    config['rand']['train']['tf_order'] = 'random'
    config['rand']['train']['rule'] = 'mean_probs'
    config['rand']['attack']['num_draws'] = 10
    config['ray']['scheduler'] = None

    device = 'cuda'
    set_random_seed(config['meta']['seed'])
    # Run main train function
    net, config, (_, _, testloader), log = main_train(config, device=device)

    # Evaluate the new model
    return_output = config['meta']['test']['save_clean_out'] or \
        config['meta']['test']['save_adv_out']
    config['rand']['use_saved_transforms'] = True
    # FIXME: Move to somewhere more explicit
    if not config['rand'].get('same_on_batch', False):
        net.module.params['test']['num_draws'] = 20
    outputs = main_test(config, net, testloader, 'test', log,
                        return_adv=config['meta']['test']['save_adv'],
                        return_output=return_output)
    log.info(f'Best trial final test accuracy: {outputs["clean"]["acc"]:.2f}.')
    log.info(f'Adv test acc: {outputs["adv"]["acc"]:.2f}')

    # Save specified outputs
    save_outputs(config, outputs)
    log.info('Finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Train a random transformation-based defense with '
                     'hyperparameter tuning via Ray Tune.'))
    parser.add_argument(
        'config_file', type=str, help='config file path')
    args = parser.parse_args()
    main(args.config_file)

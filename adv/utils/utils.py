'''Collection of utility and helper functions'''
import contextlib
import logging
import os
import pickle
import pprint
import random
from os.path import expanduser, join

import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray import tune
from torch.nn.functional import log_softmax, nll_loss, cross_entropy

from .dataset_utils import load_cifar, load_imagenet, load_pickle


@contextlib.contextmanager
def set_temp_seed(seed, devices=None):
    """Temporary sets numpy seed within a context."""
    # ====================== Save original random state ===================== #
    rand_state = random.getstate()
    np_state = np.random.get_state()
    # Get GPU devices
    if devices is None:
        num_devices = torch.cuda.device_count()
        devices = list(range(num_devices))
    else:
        devices = list(devices)
    # Save random generator state
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_states = []
    for device in devices:
        gpu_rng_states.append(torch.cuda.get_rng_state(device))
    # Set new seed
    set_random_seed(seed)
    try:
        yield
    finally:
        # Set original random state
        random.setstate(rand_state)
        # Set original numpy state
        np.random.set_state(np_state)
        # Set original torch state
        torch.set_rng_state(cpu_rng_state)
        for device, gpu_rng_state in zip(devices, gpu_rng_states):
            torch.cuda.set_rng_state(gpu_rng_state, device)


def set_random_seed(seed):
    """Set random seed for random, numpy, and torch.

    Args:
        seed (int): random seed to set
    """
    assert isinstance(seed, int)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_model_name(config):
    """Automatically get model name from config."""
    exp_id = config['meta']['exp_id']
    model_name = config['meta']['dataset'] + '_'
    if config['meta']['model_name'] is not None:
        model_name += config['meta']['model_name']
    else:
        model_name += config['meta']['method']
        if config['meta']['method'] in ['rand', 'pgd-rand']:
            model_name += '_' + '-'.join(config['rand']['transforms'])
    return model_name + '_exp' + str(exp_id)


def get_logger(name, log_dir='./', append=False, log_level=logging.DEBUG,
               console_out=True):
    """Get logger."""
    logging.basicConfig(level=log_level,
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        filename=f'{log_dir}/{name}.log',
                        filemode='a' if append else 'w')

    if console_out:
        console = logging.StreamHandler()
        console.setLevel(log_level)
        formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s')
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger().addHandler(console)

    return logging.getLogger('root')


def normalize(x, params):
    """Normalize x with given mean and std."""
    if params is None:
        return x
    if params['normalize'] is not None:
        mean = torch.tensor(params['normalize']['mean']).to(
            x.device)[(..., ) + (None, ) * (x.dim() - 2)]
        std = torch.tensor(params['normalize']['std']).to(
            x.device)[(..., ) + (None, ) * (x.dim() - 2)]
        x = (x - mean) / std
    return x


def trades_loss(clean_logits, adv_log_softmax, targets, beta):
    """Computes TRADES loss.

    Args:
        clean_logits (torch.Tensor): clean logits
        adv_log_softmax (torch.Tensor): adversarial log softmax
        targets (torch.Tensor): target labels
        beta (float): TRADES beta parameter

    Returns:
        torch.Tensor: TRADES loss
    """
    assert clean_logits.size() == adv_log_softmax.size()
    loss_natural = F.cross_entropy(clean_logits, targets, reduction='mean')
    loss_robust = F.kl_div(adv_log_softmax, F.softmax(clean_logits, dim=1),
                           reduction='batchmean')
    return loss_natural + beta * loss_robust


def load_dataset(config, dataloader=True):
    """Helper file for loading dataset.

    Args:
        config (dict): Main config dictionary
        dataloader (bool, optional): Whether to return DataLoader object or
        Dataset object. Defaults to True.

    Raises:
        NotImplementedError: Not implemented dataset.

    Returns:
        DataLoader (or Dataset): Tuple of trainset, validset, and testset
        int: number of classes
    """
    params = config['meta']
    if params['dataset'] == 'cifar10':
        load_function = load_cifar
        num_classes = 10
    elif params['dataset'] == 'cifar100':
        load_function = load_cifar
        num_classes = 100
    elif params['dataset'] == 'imagenette':
        load_function = load_imagenet
        num_classes = 10
    elif params['dataset'] == 'imagenet':
        load_function = load_imagenet
        if params['classes'] is not None:
            num_classes = len(params['classes'])
        else:
            num_classes = 1000
    else:
        raise NotImplementedError('invalid dataset.')
    params['num_classes'] = num_classes
    config['rand']['num_classes'] = num_classes

    # Load from pickled file
    if params.get('load_pickle', False):
        load_function = load_pickle

    # TODO: handle val size = 0
    assert config['meta']['val_size'] > 0

    with set_temp_seed(params['seed']):
        if dataloader:
            return load_function(load_all=False,
                                 train_batch_size=params['train']['batch_size'],
                                 test_batch_size=params['test']['batch_size'],
                                 valid_batch_size=params['valid']['batch_size'],
                                 **params
                                 ), num_classes
        return load_function(load_all=True, **params), num_classes


def select_criterion(config, mode='train'):
    # Specify loss function of the network
    if ('rand' in config['meta']['method'] and config['rand'][mode]['rule'] in
            ['mean_probs', 'majority']):
        criterion = nn.NLLLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion


def get_save_path(config, resume=False):
    if config['meta']['method'] in ['rand', 'pgd-rand']:
        rand = '-'.join(sorted(config['rand']['transforms']))
    else:
        rand = ''

    if resume:
        save_path = config['meta']['train']['resume']
    else:
        save_path = join(config['meta']['save_path'],
                         config['meta']['dataset'],
                         config['meta']['network'],
                         config['meta']['method'],
                         rand,
                         str(config['meta']['exp_id']))
    if tune.is_session_enabled():
        save_path = tune.get_trial_dir()
    save_path = expanduser(save_path)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def save_outputs(config, outputs, name='none'):
    save_path = get_save_path(config)
    filepath = f'{save_path}/save_{name}.pkl'
    pickle.dump(outputs, open(filepath, 'wb'))


def load_saved_config(config, log):
    """Retrieve saved configuration for the transformations if specified."""
    save_dir = get_save_path(config)
    if 'rand' in config['meta']['method']:
        if config['rand']['use_saved_transforms']:
            try:
                if isinstance(config['rand']['use_saved_transforms'], str):
                    save_dir = expanduser(config['rand']['use_saved_transforms'])
                else:
                    save_dir = join(save_dir, 'rand.cfg')
                saved_config = pickle.load(open(save_dir, 'rb'))
                log.info(f'Found saved config for the random transformations from {save_dir}.')
                for tf in config['rand']['transforms']:
                    config['rand'][tf] = saved_config[tf]
            except FileNotFoundError as e:
                log.info('No saved config for the random transformations.')
                log.info(f'Please specify config in {save_dir}.')
                raise e
        else:
            # If transform is not None, use the given config file
            log.info('Not use saved config for the random transformations.')
    return config


def setup_routine(config, log_name, load_config=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up model directory
    save_dir = get_save_path(config)

    # Set up logger
    log = get_logger(log_name, log_dir=save_dir,
                     log_level=config['meta'].get('log_level', logging.INFO),
                     console_out=config['meta'].get('console_out', True))
    # Pass logger to be used inside attack and/or RandWrapper
    if 'rand' in config:
        config['rand']['log'] = log
    if 'attack' in config:
        config['attack']['log'] = log

    # Load saved config
    if load_config:
        config = load_saved_config(config, log)
    log.info(f'\n{pprint.pformat(config)}')

    return config, device, save_dir, log


def compute_loss(net, criterion, outputs, targets, config, mode='test',
                 clean_inputs=None, **kwargs):
    """
    TODO: Try to include this in `forward` for each model.
    """
    assert mode in ['train', 'test']

    output_is_prob = ('rand' in config['meta']['method'] and
                      config['rand'][mode]['rule'] in ['mean_probs', 'majority'])

    if (mode == 'train' and 'pgd' in config['meta']['method'] and
            config['at']['loss_func'] in ('trades', 'mat')):
        # Compute TRADES loss
        # logits_clean = net(clean_inputs, targets=targets, mode=mode, **kwargs)
        batch_size = len(outputs) // 2
        logits_clean = outputs[:batch_size]
        logits_adv = outputs[batch_size:]
        # TODO
        adv_loss = nll_loss(logits_adv.log(), targets) if output_is_prob else cross_entropy(logits_adv, targets)
        clean_loss = nll_loss(logits_clean.log(), targets) if output_is_prob else cross_entropy(logits_clean, targets)
        loss = config['at']['beta'] * adv_loss + (1 - config['at']['beta']) * clean_loss
        # if output_is_prob:
        #     # NOTE: RandWrapper already makes sure that probability output is
        #     # positive so this should not cause any problem.
        #     loss = trades_loss(logits_clean, outputs.log(), targets, config['at']['beta'])
        # else:
        #     loss = trades_loss(logits_clean, log_softmax(outputs, dim=1), targets, config['at']['beta'])
        # TODO: Assume that output is logits
        # assert not output_is_prob
        # loss = trades_loss(logits_clean, log_softmax(outputs, dim=1), targets, config['at']['beta'])
    elif output_is_prob:
        loss = nll_loss(outputs.log(), targets)
    else:
        loss = criterion(outputs, targets)

    return loss


def compute_int(alpha, samples):
    """
    Compute `alpha`-confidence interval of `samples` with Student's 
    t-distribution.
    """
    lo, hi = st.t.interval(alpha=alpha, df=len(samples) - 1,
                           loc=np.mean(samples), scale=st.sem(samples))
    return (hi - lo) / 2


def is_rand_wrapper(model):
    from ..wrappers import RandWrapper
    return isinstance(model.module, RandWrapper)

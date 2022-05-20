import os
import pickle
import time

import numpy as np
import torch
import torch.optim as optim
from ray import tune
from torch.nn import DataParallel
from torch.optim.swa_utils import SWALR, AveragedModel

from adv.attacks import pgd_attack

from ..attacks import setup_attack
from ..models import add_normalization, create_model
from ..wrappers import create_wrapper
from .metric_utils import compute_metric
from .ray_utils import ray_report
from .test_utils import INFTY, evaluate
from .utils import (compute_loss, get_save_path, load_dataset,
                    select_criterion, set_random_seed, setup_routine)


def select_optimizer(config, net, dataloader=None):
    """Set up optimizer."""
    lr = config['meta']['train']['learning_rate']
    params = net.module.parameters() if isinstance(net, DataParallel) else net.parameters()
    if config['meta']['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.9,
                              weight_decay=config['meta']['train']['l2_reg'])
    elif config['meta']['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(params, lr=lr,
                               weight_decay=config['meta']['train']['l2_reg'])
    else:
        raise NotImplementedError('Optimizer not implemented.')
    return optimizer


def select_lr_scheduler(config, optimizer):
    """Set up learning rate schedule."""
    lr = config['meta']['train']['learning_rate']
    epochs = config['meta']['train']['epochs']

    if config['meta']['train']['lr_scheduler'] == 'cyclic':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, lr, epochs=epochs, steps_per_epoch=422,
            pct_start=0.5, anneal_strategy='linear', cycle_momentum=False,
            base_momentum=0.9, div_factor=1e5, final_div_factor=1e5)
    elif config['meta']['train']['lr_scheduler'] == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config['meta']['train']['lr_steps'], gamma=0.1)
    elif config['meta']['train']['lr_scheduler'] == 'step-half':
        # Reduce learning rate by half every 10 epochs
        step_len = config['meta']['train']['step_len']
        steps = np.arange(step_len, epochs, step_len)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=0.5)
    elif config['meta']['train']['lr_scheduler'] == 'cyclic-cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config['meta']['train']['step_len'], T_mult=2, eta_min=1e-5)
    elif config['meta']['train']['lr_scheduler'] == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    else:
        lr_scheduler = None
    return lr_scheduler


def set_curriculum(config, epoch):
    """Set difficulty for curriculum learning."""
    if config['at'].get('step_gap'):
        # Increase probability gap in steps when using ATES
        if epoch in config['at']['step_gap']:
            config['at']['gap'] += ((config['at']['final_gap'] - config['at']['init_gap'])
                                    / len(config['at']['step_gap']))
    elif config['at'].get('linear_gap'):
        # Increase probability gap linearly when using ATES
        lin_gap = config['at']['linear_gap']
        interval = lin_gap[1] - lin_gap[0]
        if lin_gap[0] <= epoch < lin_gap[1]:
            config['at']['gap'] += (config['at']['final_gap'] - config['at']['init_gap']) / interval

    # calculate fosc threshold if Dynamic AT is used
    if config['at'].get('use_fosc'):
        fosc_thres = config['at']['fosc_max'] * \
            (1 - (epoch / config['at']['dynamic_epoch']))
        config['at']['fosc_thres'] = np.maximum(0, fosc_thres)

    return config


def validate(config, net, criterion, validloader, attacks_dict, adv, rand):
    # Compute loss and accuracy on validation set
    val_samples = config['meta']['valid'].get('num_samples', INFTY)
    clean_out = evaluate(net, validloader, criterion, config,
                         rand=rand, adv=False, num_samples=val_samples)

    # Validate on adversarial examples every `save_epochs` if specified
    adv_out = None
    if adv or ('attack' in attacks_dict):
        if 'attack' in attacks_dict:
            attack = attacks_dict['attack']
        else:
            attack = [attacks_dict['at']]
        adv_out = evaluate(net, validloader, criterion, config, rand=rand,
                           adv=adv, attacks=attack, num_samples=val_samples)
    return clean_out, adv_out

# =========================================================================== #
#                           One-epoch train function                          #
# =========================================================================== #


def train(net, trainloader, validloader, criterion, optimizer, config,
          epoch, log, best_metric=0, lr_scheduler=None, adv=True,
          attacks_dict=None, rand=True):
    """Train `net` for one epoch."""
    net.train()
    train_loss, train_correct, train_total = 0, 0, 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = get_save_path(config)
    train_params = config['meta']['train']
    num_expand = config['meta']['train'].get('expand_inputs', 1)
    repeat = True
    num_steps = len(trainloader)

    for i, (inputs, targets) in enumerate(trainloader):
        start = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        if num_expand > 1:
            inputs = inputs.repeat_interleave(num_expand)
            targets = targets.repeat_interleave(num_expand)
        optimizer.zero_grad()
        # Run adversarial training if specified
        if adv:
            net.eval()
            x = attacks_dict['at'].attack_batch(inputs, targets).detach()
            repeat = not config['at'].get('mult_pert', False)
            net.train()
            optimizer.zero_grad()
        else:
            x = inputs
        # Pass training samples to the model
        # DEBUG
        # start = time.time()
        if config['at']['loss_func'] in ('trades', 'mat') and adv:
            # TODO: support rand mode
            x_ = torch.cat([inputs, x], dim=0)
            output = net(x_, mode='train')
            loss = compute_loss(net, criterion, output, targets, config,
                                rand=rand, mode='train')
            output = output[inputs.size(0):]
            # output = net(x, targets=targets, mode='train', rand=rand, repeat=repeat)
            # loss = compute_loss(net, criterion, output, targets, config,
            #                     rand=rand, mode='train', clean_inputs=inputs)
        else:
            output = net(x, targets=targets, mode='train', rand=rand, repeat=repeat)
            loss = compute_loss(net, criterion, output, targets, config,
                                rand=rand, mode='train')
        # print(time.time() - start)
        loss.backward()
        optimizer.step()
        if train_params['lr_scheduler'] == 'cyclic':
            lr_scheduler.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = output.max(1)
        train_total += inputs.size(0)
        train_correct += predicted.eq(targets).float().sum().item()

        if i % 50 == 0:
            acc = predicted.eq(targets).float().mean().item()
            log.info(f'  step {i}/{num_steps}, time: {time.time() - start:.4f},'
                     f' loss: {loss.item():.4f}, acc: {acc:.2f}')

    clean_out, adv_out = validate(config, net, criterion, validloader, attacks_dict, adv, rand)
    message = (f'{epoch:6d} | {train_loss / train_total:6.4f}, '
               f'{train_correct / train_total * 100:5.2f} | '
               f'{clean_out["loss"]:6.4f}, {clean_out["acc"]:5.2f} | ')
    if adv_out is not None:
        message += f'{adv_out["loss"]:6.4f}, {adv_out["acc"]:5.2f} |'
        adv_acc = adv_out["acc"]
    else:
        message += '  -   ,   -   |'
        adv_acc = 0
    log.info(message)

    # Compute metric for saving best model
    track_metric = compute_metric(train_params['metric'], clean_out['acc'], adv_acc)

    # Call tune report when scheduler is used
    if (tune.is_session_enabled() and config['ray']['scheduler'] is not None
            and (epoch + 1) % config['ray']['metric']['report_epochs'] == 0):
        ray_report(config, clean_out['acc'], adv_acc)

    # Save model weights
    if not train_params['save_best_only']:
        # Save model every `save_epochs` epochs
        if (epoch + 1) % train_params['save_epochs'] == 0:
            log.info('Saving model...')
            if hasattr(net, 'module'):
                net.module.save_weights(f'{save_dir}/model_epoch{epoch}.pt')
            else:
                net.save_weights(f'{save_dir}/model_epoch{epoch}.pt')
    if track_metric > best_metric:
        # Save the best model according to `metric`
        log.info('Saving best model...')
        if hasattr(net, 'module'):
            net.module.save_weights(f'{save_dir}/model.pt')
        else:
            net.save_weights(f'{save_dir}/model.pt')
        best_metric = track_metric
    return best_metric, track_metric


# =========================================================================== #
#                              Main train function                            #
# =========================================================================== #
def main_train(config, device=None, load_config=True):
    """Main training function.

    Args:
        config (dict): Main config
        device (str, optional): device to put model on (e.g., 'cuda', 'cuda:0')

    Returns:
        net: Trained best network
        config: Updated main config
        dataloaders: DataLoaders
        log: Logger
    """
    # Training parameters
    rand = 'rand' in config['meta']['method']
    adv = 'pgd' in config['meta']['method']

    # Set all random seeds
    set_random_seed(config['meta']['seed'])

    # Initialize save directory and logging
    config, device_temp, save_dir, log = setup_routine(config, 'train',
                                                       load_config=load_config)
    if device is None:
        device = device_temp

    # Load dataset
    log.info('Preparing data...')
    (trainloader, validloader, testloader), num_classes = load_dataset(
        config, dataloader=True)

    # Build neural network
    log.info('Building model...')
    base_net = create_model(config, num_classes)
    base_net = add_normalization(base_net, config['meta'])
    # Wrap the neural network with module for random transformations or
    # adversarial training
    net = create_wrapper(base_net, config, 'train', device=device)
    log.info(net)
    net_module = net.module if hasattr(net, 'module') else net

    # Load pretrained weights if specified
    weight_path = config['meta'].get('pretrained')
    if isinstance(weight_path, str):
        weight_path = os.path.expanduser(weight_path)
        print(f'Loading weights from {weight_path}...')
        if not os.path.exists(weight_path):
            raise FileNotFoundError('Model does not exist. Weights are randomly initialized.')
        net_module.load_weights(weight_path)

    # Save the config for future testing purposes
    if rand:
        pickle.dump(config['rand'], open(f'{save_dir}/rand.cfg', 'wb'))

    criterion = select_criterion(config, mode='train')
    optimizer = select_optimizer(config, net, dataloader=trainloader)
    lr_scheduler = select_lr_scheduler(config, optimizer)
    attacks_dict = setup_attack(config, net, log, 'train')

    train_params = config['meta']['train']
    epochs = train_params['epochs']
    swa = train_params.get('swa', False)
    swa_start = train_params.get('swa_start', int(epochs * 0.75))
    if swa:
        swa_model = AveragedModel(net)
        swa_scheduler = SWALR(optimizer,
                              swa_lr=train_params.get('swa_lr', 1e-3),
                              anneal_epochs=10)

    # ======================================================================= #
    #                           Start Training loop                           #
    # ======================================================================= #
    log.info(' epoch | loss  , acc   | val_l , val_a | adv_l , adv_a |')
    best_metric = - float('inf')
    patience = 20
    current_best_metric = - float('inf')
    num_down_epochs = 0
    for epoch in range(epochs):
        # Main training function
        best_metric, metric = train(
            net, trainloader, validloader, criterion, optimizer, config, epoch,
            log, best_metric=best_metric, lr_scheduler=lr_scheduler,
            attacks_dict=attacks_dict, adv=adv, rand=rand)

        if swa and epoch > swa_start:
            swa_model.update_parameters(net)
            swa_scheduler.step()
            torch.optim.swa_utils.update_bn(trainloader, swa_model)
            clean_out, adv_out = validate(config, swa_model, criterion, validloader, attacks_dict, adv, rand)
            message = (f'                swa acc: '
                       f'{clean_out["loss"]:6.4f}, {clean_out["acc"]:5.2f} | '
                       f'{adv_out["loss"]:6.4f}, {adv_out["acc"]:5.2f} |')
            log.info(message)
            # TODO: save best for SWA?
            swa_model.module.module.save_weights(f'{save_dir}/swa_model_epoch{epoch}.pt')
        else:
            if train_params['lr_scheduler'] in ['step', 'step-half', 'cos']:
                lr_scheduler.step()

        # Terminate early if no progress is made
        if not swa:
            if metric < current_best_metric:
                num_down_epochs += 1
            else:
                num_down_epochs = 0
                current_best_metric = metric

        if num_down_epochs >= patience:
            break

    # ======================================================================= #
    #                            End Training loop                            #
    # ======================================================================= #
    if swa:
        net_module.load_weights(f'{save_dir}/swa_model_epoch{epoch}.pt')
    else:
        net_module.load_weights(f'{save_dir}/model.pt')
    return net, config, (trainloader, validloader, testloader), log

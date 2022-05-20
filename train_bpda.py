'''Train BPDA model'''
from __future__ import print_function

import argparse
import os
import pprint

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from adv.wrappers.transform import get_transform, init_transform, name_to_init_tf
from adv.models.bpda import parse_bpda_name, get_bpda_network
from adv.utils import (get_logger, set_random_seed,
                       load_dataset, parse_model_name)


def forward(transform, net, criterion, x):
    x_new, params = transform(x)
    x_pred = net(x, params)
    loss = criterion(x_pred, x_new)
    return x_pred, loss


def evaluate(transform, net, dataloader, criterion, config, device):
    """Evaluate network."""

    net.eval()
    val_cum_loss = 0
    val_total = 0

    with torch.no_grad():
        for batch_idx, (x, targets) in enumerate(dataloader):
            x = x.to(device)
            size = x.shape[0]

            _, loss = forward(transform, net, criterion, x)

            val_cum_loss += loss.item()
            val_total += size

    return val_cum_loss / val_total


def train(transform, net, trainloader, validloader, criterion, optimizer,
          config, epoch, device, log, best_loss, model_path, lr_scheduler=None,
          log_freq=1000):
    """Main training function."""

    row_coords = None
    col_coords = None

    net.train()
    train_cum_loss = 0
    train_total = 0

    for batch_idx, (x, targets) in enumerate(trainloader):
        x = x.to(device)
        size = x.shape[0]

        optimizer.zero_grad()

        _, loss = forward(transform, net, criterion, x)

        train_cum_loss += loss.item()
        train_total += size
        loss.backward()
        optimizer.step()
        if batch_idx % log_freq == log_freq - 1:
            train_loss = train_cum_loss / train_total
            train_cum_loss = 0
            train_total = 0
            log.info(f"Train loss: {train_loss}")

    train_loss = train_cum_loss / train_total
    val_loss = evaluate(transform, net, validloader, criterion, config,
                        device)

    log.info(f"Train loss: {train_loss}, Valid loss: {val_loss}")

    state_dict = net.state_dict()
    if not config['meta']['save_best_only']:
        # Save model every <save_epochs> epochs
        if epoch % config['meta']['save_epochs'] == 0:
            log.info('Saving model...')
            torch.save(state_dict, model_path + '_epoch%d.pt' % epoch)
    elif config['meta']['save_best_only'] and val_loss < best_loss:
        # Save only the model with the highest adversarial accuracy
        log.info('Saving model...')
        torch.save(state_dict, model_path + '.pt')
        best_loss = val_loss

    return best_loss


def main(config_file):
    """Main function. Use config file train_bpda.yml"""

    # Parse config file
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config['meta']['gpu_id']

    # Training parameters
    epochs = config['meta']['epochs']
    lr = config['meta']['learning_rate']
    tf = config['transform']
    config[tf]['input_size'] = 224
    if tf in name_to_init_tf:
        module = init_transform(tf, config[tf])
    else:
        module = None
    transform = get_transform(tf, config[tf], module=module)

    # Set all random seeds
    set_random_seed(config['meta']['seed'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up model directory
    bpda_name = parse_bpda_name(config)
    save_dir = os.path.join(
        config['meta']['save_path'], 'saved_models', bpda_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, 'model')
    model_params_path = os.path.join(save_dir, 'transform.yml')
    with open(model_params_path, 'w') as file:
        yaml.dump(config, file)

    # Set up logger
    log = get_logger(bpda_name, 'logs')
    log.info('\n%s', pprint.pformat(config))

    # Load dataset
    log.info('Preparing data...')
    (trainloader, validloader, testloader), num_classes = load_dataset(config, 'train')

    # Build neural network
    log.info('Building model...')
    net = get_bpda_network(config[tf])
    net = net.to(device)

    # If GPU is available, allows parallel computation and cudnn speed-up
    # if device == 'cuda':
    #    net = nn.DataParallel(net)
    #    cudnn.benchmark = True

    # Specify loss function of the network
    criterion = nn.MSELoss()

    # Set up optimizer
    if config['meta']['optimizer'] == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=config['meta']['l2_reg'])
    elif config['meta']['optimizer'] == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr,
                               weight_decay=config['meta']['l2_reg'])
    else:
        raise NotImplementedError('Optimizer not implemented.')

    # Set up learning rate schedule
    if config['meta']['lr_scheduler'] == 'cyclic':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, lr, epochs=epochs, steps_per_epoch=422,
            pct_start=0.5, anneal_strategy='linear', cycle_momentum=False,
            base_momentum=0.9, div_factor=1e5, final_div_factor=1e5)
    elif config['meta']['lr_scheduler'] == 'step':
        if epochs <= 70:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [40, 50, 60], gamma=0.1)
        elif epochs <= 100:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [40, 60, 80], gamma=0.1)
        elif epochs <= 160:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [60, 80, 100, 120, 140], gamma=0.2)
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [60, 120, 160], gamma=0.2)
    else:
        lr_scheduler = None

    # Starting the main training loop over epochs
    best_loss = np.inf
    for epoch in range(epochs):
        best_loss = train(transform, net, trainloader, validloader, criterion,
                          optimizer, config, epoch, device, log, best_loss,
                          model_path, lr_scheduler=lr_scheduler)

        if config['meta']['lr_scheduler'] == 'step':
            lr_scheduler.step()

    # Evaluate network on clean data
    test_loss = evaluate(transform, net, testloader, criterion, config, device)
    log.info('Test loss: %.4f', test_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BPDA')
    parser.add_argument(
        'config_file', type=str, help='name of config file')
    args = parser.parse_args()
    main(args.config_file)

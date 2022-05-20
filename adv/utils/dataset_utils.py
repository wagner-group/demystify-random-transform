'''
Dataset and DataLoader adapted from
https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
'''

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def load_pickle(data_path='./data', test_batch_size=128, shuffle=False,
                num_workers=4, **kwargs):
    """Load MNIST data into train/val/test data loader"""

    data = pickle.load(open(data_path, 'rb'))['adv']['x_adv']
    if not isinstance(data, tuple):
        input_size = 224
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
        # TODO: remove this fix in future versions
        clean_testset = torchvision.datasets.ImageFolder(
            '~/data/imagenette2-320/val', transform=transform)
        raise NotImplementedError

    testset = torch.utils.data.TensorDataset(
        torch.from_numpy(data[0][0]).float(), torch.from_numpy(data[1]).long())
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=True, pin_memory=True)

    return None, None, testloader

# ============================================================================ #


def load_cifar(dataset='cifar10', data_path='./data', train_batch_size=128,
               test_batch_size=128, valid_batch_size=128, load_all=True,
               val_size=0.1, augment=True, shuffle=True, num_workers=4,
               **kwargs):
    """Load CIFAR-10/100 data into train/val/test data loader"""

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor()
        ])
    else:
        transform_train = transform

    if dataset == 'cifar10':
        dataset_func = torchvision.datasets.CIFAR10
    elif dataset == 'cifar10':
        dataset_func = torchvision.datasets.CIFAR100
    else:
        raise NotImplementedError(
            'Invalid dataset (options: cifar10, cifar100)')

    trainset = dataset_func(
        root=data_path, train=True, download=True, transform=transform_train)
    validset = dataset_func(
        root=data_path, train=True, download=True, transform=transform)
    testset = dataset_func(
        root=data_path, train=False, download=True, transform=transform)

    # Random split train and validation sets and shuffle them if needed
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))
    test_idx = list(range(len(testset)))
    if shuffle:
        np.random.shuffle(indices)
        np.random.shuffle(test_idx)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    if load_all:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=(num_train - split), sampler=train_sampler)
        validloader = torch.utils.data.DataLoader(
            validset, batch_size=split, shuffle=False, sampler=valid_idx)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset), shuffle=True)

        x_train = next(iter(trainloader))
        x_valid = next(iter(validloader))
        x_test = next(iter(testloader))
        return x_train, x_valid, x_test

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, drop_last=True,
        sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=valid_batch_size, shuffle=False, drop_last=True,
        sampler=valid_idx, num_workers=num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, drop_last=True,
        sampler=test_idx, num_workers=num_workers, pin_memory=True)
    return trainloader, validloader, testloader

# ============================================================================ #


def load_imagenet(data_path='./data', train_batch_size=128,
                  test_batch_size=128, valid_batch_size=128, load_all=True,
                  val_size=0.1, augment=True, shuffle=True, classes=None,
                  num_workers=4, **kwargs):
    """Load ImagetNet data into train/val/test data loader"""

    input_size = 224

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transform

    trainset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, 'train'), transform=transform_train)
    validset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, 'train'), transform=transform)
    testset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, 'val'), transform=transform)

    # Get subset of classes if specified
    if classes is not None:
        relabel(trainset, classes)
        relabel(validset, classes)
        relabel(testset, classes)

    # Random split train and validation sets and shuffle them if needed
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))
    test_idx = list(range(len(testset)))
    if shuffle:
        np.random.shuffle(indices)
        np.random.shuffle(test_idx)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)

    if load_all:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=(num_train - split), sampler=train_sampler)
        validloader = torch.utils.data.DataLoader(
            validset, batch_size=split, shuffle=False, sampler=valid_idx)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset), shuffle=False, sampler=test_idx)

        x_train = next(iter(trainloader))
        x_valid = next(iter(validloader))
        x_test = next(iter(testloader))
        return x_train, x_valid, x_test

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, drop_last=True,
        sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=valid_batch_size, shuffle=False, drop_last=True,
        sampler=valid_idx, num_workers=num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, drop_last=True,
        sampler=test_idx, num_workers=num_workers, pin_memory=True)
    return trainloader, validloader, testloader


def get_label_idx(dataset, classes):
    targets = torch.tensor(dataset.targets)
    indices = []
    for label in classes:
        indices.append((targets == label).nonzero())
    return torch.cat(indices)


def relabel(dataset, classes):
    new_list = []
    for input, target in dataset.samples:
        try:
            new_target = classes.index(target)
            new_list.append((input, new_target))
        except ValueError:
            pass
    dataset.samples = new_list

# ============================================================================ #


def create_spheres(d=500, num_total=1e7, radii=(1, 1.3), centers=(0, 0),
                   test_size=0.2, val_size=0.1, seed=1):
    """
    Create sphere dataset: two spheres in space R^d with the specified radii
    """

    np.random.seed(seed)
    num_total = int(num_total)

    # Samples R^d vectors from a normal distribution and normalize them
    spheres = torch.randn((num_total, int(d)))
    spheres = F.normalize(spheres, 2, 1)
    # Scale 1st and 2nd halves of the vectors to the 1st and 2nd radii
    spheres[:num_total // 2] *= radii[0]
    spheres[num_total // 2:] *= radii[1]
    # Shifting first dim to centers
    spheres[:num_total // 2] += centers[0]
    spheres[num_total // 2:] += centers[1]
    # Expand to 4 dim
    spheres = spheres.view(num_total, 1, 1, int(d))

    indices = np.arange(num_total)
    np.random.shuffle(indices)

    train_idx = int(num_total * (1 - test_size - val_size))
    test_idx = int(num_total * (1 - test_size))
    x_train = spheres[indices[:train_idx]]
    x_valid = spheres[indices[train_idx:test_idx]]
    x_test = spheres[indices[test_idx:]]

    y_train = torch.tensor(
        (indices[:train_idx] >= num_total // 2).astype(np.int64))
    y_valid = torch.tensor(
        (indices[train_idx:test_idx] >= num_total // 2).astype(np.int64))
    y_test = torch.tensor(
        (indices[test_idx:] >= num_total // 2).astype(np.int64))

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def load_spheres(batch_size=100, load_all=True, d=500, num_total=1e7,
                 radii=(1, 1.3), centers=(0, 0), test_size=0.2, val_size=0.1,
                 shuffle=True, seed=1, **kwargs):

    num_workers = 4

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = create_spheres(
        d=d, num_total=num_total, radii=radii, centers=centers,
        test_size=test_size, val_size=val_size, seed=seed)

    if load_all:
        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    validset = torch.utils.data.TensorDataset(x_valid, y_valid)
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers)
    return trainloader, validloader, testloader

# ============================================================================ #

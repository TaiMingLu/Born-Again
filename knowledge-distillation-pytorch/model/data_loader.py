"""
Dataset utilities and data loaders for CIFAR variants.
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

_DATASETS = {
    'cifar10': {
        'dataset_cls': torchvision.datasets.CIFAR10,
        'root': './data-cifar10',
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2470, 0.2435, 0.2616),
        'num_classes': 10,
    },
    'cifar100': {
        'dataset_cls': torchvision.datasets.CIFAR100,
        'root': './data-cifar100',
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'num_classes': 100,
    }
}


def get_dataset_info(name):
    """Return metadata for a supported dataset."""
    key = name.lower()
    if key not in _DATASETS:
        raise ValueError("Unsupported dataset '{}'. Available: {}".format(
            name, ", ".join(sorted(_DATASETS.keys()))))
    return _DATASETS[key]


def _resolve_transforms(params, mean, std):
    """Create train/dev transforms with optional augmentation."""
    normalize = transforms.Normalize(mean, std)
    if getattr(params, 'augmentation', "yes") == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    return train_transformer, dev_transformer


def _build_datasets(dataset_key, train_transform, dev_transform, params):
    """Instantiate torchvision datasets with project defaults."""
    info = get_dataset_info(dataset_key)
    data_root = getattr(params, 'data_root', info['root'])
    dataset_cls = info['dataset_cls']

    trainset = dataset_cls(root=data_root, train=True, download=True, transform=train_transform)
    devset = dataset_cls(root=data_root, train=False, download=True, transform=dev_transform)
    return trainset, devset

def fetch_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """

    dataset_key = getattr(params, 'dataset', 'cifar10')
    info = get_dataset_info(dataset_key)
    train_transformer, dev_transformer = _resolve_transforms(params, info['mean'], info['std'])

    trainset, devset = _build_datasets(dataset_key, train_transformer, dev_transformer, params)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(
        devset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl


def fetch_subset_dataloader(types, params):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """

    dataset_key = getattr(params, 'dataset', 'cifar10')
    info = get_dataset_info(dataset_key)
    train_transformer, dev_transformer = _resolve_transforms(params, info['mean'], info['std'])

    trainset, devset = _build_datasets(dataset_key, train_transformer, dev_transformer, params)

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl

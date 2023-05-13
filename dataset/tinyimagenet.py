from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision import transforms
from PIL import Image

def get_data_folder(dataset='tinyimagenet'):
    """
    return the path to store the data
    """
    data_folder = '../../data/tiny-imagenet-200/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class TinyImageInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

def get_tinyimagenet_dataloader(dataset='tinyimagenet',
                                batch_size=128,
                                num_workers=16,
                                is_instance=False,
                                multiprocessing_distributed=False):
    """
        Data Loader for imagenet
        """
    if dataset == 'tinyimagenet':
        data_folder = get_data_folder(dataset)
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # dataset
    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'val')
    if is_instance:
        train_set = TinyImageInstance(root=train_folder,
                                     train=True,
                                     transform=train_transforms)
        n_data = len(train_set)
    else:
        train_set = TinyImageInstance(root=train_folder,
                                      train=True,
                                      transform=train_transforms)

    test_set = TinyImageInstance(test_folder, transform=val_transforms)

    # dataloader
    if multiprocessing_distributed:
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(test_set, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              sampler=train_sampler)

    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=True,
                             num_workers=int(num_workers/2),
                             pin_memory=True,
                             sampler=val_sampler)
    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


import os
import numpy as np
from PIL.Image import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision import transforms



class STL10BackCompat(datasets.STL10):
    """
    STL10Instance+Sample Dataset
    """

    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    @property
    def train_data(self):
        return self.data

    @property
    def test_data(self):
        return self.data


class STL10Instance(STL10BackCompat):
    """STL10Instance Dataset"""

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_stl10_dataloader(dataset='stl10', batch_size=128, num_workers=8, is_instance=False):
    """
    stl 10
    """
    if dataset == 'stl10':
        data_folder = get_data_folder(dataset)
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
    ])

    if is_instance:
        train_set = STL10Instance(root=data_folder,
                                  download=True,
                                  split='train',
                                  transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.STL10(root=data_folder,
                                   download=True,
                                   split='train',
                                   transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.STL10(root=data_folder,
                              download=True,
                              split='test',
                              transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size / 2),
                             shuffle=False,
                             num_workers=int(num_workers / 2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


def get_data_folder(dataset='stl10'):
    """
    return the path to store the data
    """
    # data_folder = '../../../wangbo/data/tiny-imagenet-200'
    data_folder = '../../data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


def get_stl10_dataloaders(dataset='stl10',
                          batch_size=128,
                          num_workers=8,
                          multiprocessing_distributed=False):
    """
    cifar 100
    """
    if dataset == 'stl10':
        data_folder = get_data_folder(dataset)
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.Normalize(mean=[0.431, 0.415, 0.366], std=[0.268, 0.261, 0.268])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.Normalize(mean=[0.431, 0.415, 0.366], std=[0.268, 0.261, 0.268])
    ])


    train_set = datasets.STL10(root=data_folder,
                               split='train',
                               transform=train_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.STL10(root=data_folder,
                              split='test',
                              transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader

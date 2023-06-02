from __future__ import print_function

import os
import sys

import numpy as np
from torch.utils.data import Dataset, DataLoader
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


class TinyImage(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform=None,
            target_transform=None
    ) -> None:
        super(TinyImage, self).__init__(root,
                                        transform=transform,
                                        target_transform=target_transform,
                                        )
        pass
        # self.data = []
        # self.targets = []
        # for index in range(len(self.imgs)):
        #     path, target = self.imgs[index]
        #     img = self.loader(path)
        #     self.data.append(img)
        #     self.targets.append(target)
        #     self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        #     self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.imgs[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.loader(img)
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class TinyImageBackCompat(TinyImage):
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


class TinyImageNetInstanceSample(TinyImageBackCompat):
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = len(self.data)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_tinyimagenet_dataloaders_sample(dataset='tinyimagenet',
                                        batch_size=128, num_workers=16,
                                        k=4096, mode='exact',
                                        is_sample=True, percent=1.0,
                                        multiprocessing_distributed=False):
    """
    tinyimagenet
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

    train_set = TinyImageNetInstanceSample(root=data_folder,
                                           train=True,
                                           transform=train_transforms,
                                           k=k,
                                           mode=mode,
                                           is_sample=is_sample,
                                           percent=percent)
    n_data = len(train_set)
    test_set = TinyImageNet_load(data_folder, train=False, transform=val_transforms)

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
                             batch_size=int(batch_size / 2),
                             shuffle=True,
                             num_workers=int(num_workers / 2),
                             pin_memory=True,
                             sampler=val_sampler)
    return train_loader, test_loader, n_data


class TinyImageNet_load(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


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
    if is_instance:
        train_set = TinyImageNet_load(root=data_folder,
                                      train=True,
                                      transform=train_transforms)
        n_data = len(train_set)
    else:
        train_set = TinyImageNet_load(root=data_folder,
                                      train=True,
                                      transform=train_transforms)

    test_set = TinyImageNet_load(data_folder, train=False, transform=val_transforms)

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
                             batch_size=int(batch_size / 2),
                             shuffle=True,
                             num_workers=int(num_workers / 2),
                             pin_memory=True,
                             sampler=val_sampler)
    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader

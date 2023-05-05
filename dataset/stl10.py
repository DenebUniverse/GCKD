import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision import transforms

def get_data_folder(dataset='stl10'):
    """
    return the path to store the data
    """
    data_folder = '../../data/stl10'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class STL10Instance(datasets.STL10):
    """STL10 datasets which returns the index of the image as well"""
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        img = np.transpose(img, (1, 2, 0)) # convert to (H, W, C) from (C, H, W)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class STL10Sample(datasets.STL10):
    """: STL10 datasets which returns (img, label, index, contrast_index):
    """
    def __init__(self, root, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(root=root, split='train', transform=transform,
                         target_transform=target_transform, download=True)
        self.k = k
        self.is_sample = is_sample

        if self.is_sample:
            num_classes = len(self.classes)
            num_samples = len(self.data)
            label = self.labels

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

        print('dataset initialized!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        img = np.transpose(img, (1, 2, 0)) # convert to (H, W, C) from (C, H, W)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index

def get_test_loader(dataset='stl10', batch_size=128, num_workers=8):
    """get the test data loader"""

    if dataset == 'stl10':
        data_folder = '../../../wangbo/data/stl10_binary'
        test_set = datasets.STL10(data_folder, split='test', transform=transforms.ToTensor(), download=True)
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    return test_loader


class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class ImageFolderSample(datasets.STL10):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """
    def __init__(self, root, transform=None, target_transform=None,
                 is_sample=False, k=4096, **kwargs):
        super().__init__(root=root, transform=transform, target_transform=target_transform, **kwargs)

        self.k = k
        self.is_sample = is_sample

        if self.is_sample:
            num_classes = len(self.classes)
            num_samples = len(self.data)
            label = self.labels

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            # sample_idx = np.hstack((neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index

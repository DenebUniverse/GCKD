"""
We transfer the representation learned from CIFAR100 to STL-10 and TinyImageNet datasets
by freezing the network and training a linear classifier on top of the last feature layer
(the layer prior to the logit)
to perform 10-way (STL-10) or 200-way (TinyImageNet) classification.
We then train a linear classifier to perform 10-way (for STL-10)
or 200-way (for TinyImageNet) classification(all images downsampled to 32x32)
to quantify the transferability of the representations.
For this experiment, we use the combination of teacher network WRN-40-2
and student network WRN-16-2. Classification accuracies (%) are reported.

"""


import io
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

# Define the transformations to apply to the images
from torch.utils.data import DataLoader
from models import model_dict

from helper.util import AverageMeter, accuracy, reduce_tensor


def get_stl10_dataloaders(batch_size=64, num_workers=8):
    """
    stl10 100
    """
    transform = transforms.Compose([
        transforms.Resize(32),
        # transforms.CenterCrop(32),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.431, 0.415, 0.366], std=[0.268, 0.261, 0.268])
    ])

    # dataset
    stl10_dataset_train = datasets.STL10(root='../../data/',
                                   split='train',
                                   download=True,
                                   transform=transform)

    train_loader = DataLoader(stl10_dataset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    stl10_dataset_test = datasets.STL10(root='../../data/',
                                   split='test',
                                   download=True,
                                   transform=transform)

    test_loader = DataLoader(stl10_dataset_test,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, stl10_dataset_train


def get_tinyimagenet_dataloaders(batch_size=64, num_workers=8):
    """
    tinyimagenet
    """
    transform = transforms.Compose([
        transforms.Resize(32),
        # transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset
    tinyimagenet_train = datasets.ImageFolder(root='../../data/tiny-imagenet-200/train',
                                                transform=transform)

    train_loader = DataLoader(tinyimagenet_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    tinyimagenet_test = datasets.ImageFolder(root='../../data/tiny-imagenet-200/test',
                                                transform=transform)

    test_loader = DataLoader(tinyimagenet_test,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, tinyimagenet_train


def transfer_resnet():
    '''
        resnet8x4_100 ->
    '''

    try:
        model_s = model_dict['resnet8x4_100'](num_classes=100)
    except KeyError:
        print("This model is not supported.")

    # model_s
    student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_gld-G_TAG_momentum-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_1.0-m_1.0-b_3.0-lr_None-clT_0.07-kdT_4-2008_1/resnet8x4_best.pth'

    # Load the model state dictionary from a file into a buffer
    with open(student_path, 'rb') as f:
        buffer = io.BytesIO(f.read())

    checkpoint = torch.load(buffer)
    model_s.load_state_dict(checkpoint['model'])

    for param in model_s.parameters():
        param.requires_grad = False

    # Define the linear classifier to train on top of the last feature layer
    num_classes_stl10 = 10
    num_classes_tinyimagenet = 200
    in_features = model_s.fc.in_features  # 256

    # classifier_stl10 = nn.Linear(in_features, num_classes_stl10)    # 256 -> 10

    model_s.fc = nn.Linear(in_features, num_classes_stl10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_s = model_s.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_stl10 = torch.optim.SGD(model_s.fc.parameters(), lr=0.1, momentum=0.9)

    # Train the linear classifier on the STL-10 dataset
    stl10_train_loader, stl10_test_loader, stl10_dataset_train = get_stl10_dataloaders(batch_size=64, num_workers=8)
    for epoch in range(10):
        running_loss = 0.0

        for idx, batch_data in enumerate(stl10_train_loader):
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model_s(inputs)
            loss = criterion(outputs, labels)

            # ===================backward=====================
            optimizer_stl10.zero_grad()
            loss.backward()
            optimizer_stl10.step()
            running_loss += loss.item() * inputs.size(0)

            # if idx % 200 == 0:
            #     print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / len(stl10_dataset_train)))
            if idx % 100 == 0:  # Print every 100 mini-batches
                print('[Epoch %d, Batch %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 100))

    # Test the linear classifier on the STL-10 dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, batch_data in enumerate(stl10_test_loader):
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model_s(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if idx % 200 == 0:
                print('STL-10 accuracy: %d %%' % (100 * correct / total))


def tranfer_monilenet():
    """
        MobileNetV2_100
    """

    try:
        model_s = model_dict['MobileNetV2_100'](num_classes=100)
    except KeyError:
        print("This model is not supported.")

    student_path = './save/students/models/S_MobileNetV2-T_ResNet50-D_cifar100_64-M_gckd_1-G_TAG_2_momentum-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_1.0-m_1.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-797_1/MobileNetV2_best.pth'
    # student_path = './save/students/models/S_MobileNetV2-T_ResNet50-D_cifar100_64-M_gckd_1-G_TAG_1_one-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_1.0-m_1.0-b_3.0-lr_None-clT_0.07-kdT_4-1057_1/MobileNetV2_best.pth'

    # Load the model state dictionary from a file into a buffer
    with open(student_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    checkpoint = torch.load(buffer)
    model_s.load_state_dict(checkpoint['model'])

    for param in model_s.parameters():
        param.requires_grad = False

    num_classes_stl10 = 10

    in_features = model_s.classifier[0].in_features     # 1280

    # Define the linear classifier to train on top of the last feature layer
    model_s.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(in_features, num_classes_stl10),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_s = model_s.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_stl = torch.optim.SGD(model_s.classifier.parameters(), lr=0.1, momentum=0.9)

    stl10_train_loader, stl10_test_loader, stl10_train = get_stl10_dataloaders(batch_size=64, num_workers=8)

    # Train the linear classifier on the TinyImageNet dataset
    arr=[]
    for epoch in range(10):
        running_loss = 0.0

        model_s.train()
        for idx, batch_data in enumerate(stl10_train_loader):
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model_s(inputs)
            loss = criterion(outputs, labels)

            # ===================backward=====================
            optimizer_stl.zero_grad()
            loss.backward()
            optimizer_stl.step()

            running_loss += loss.item() * inputs.size(0)

            # if idx % 200 == 0:
            #     print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / len(stl10_train)))

        # Test the linear classifier on the STL-10 dataset

        correct = 0
        total = 0
        test_acc = AverageMeter()

        model_s.eval()
        with torch.no_grad():
            for idx, batch_data in enumerate(stl10_test_loader):
                inputs, labels = batch_data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model_s(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                metrics = accuracy(outputs, labels, topk=(1, 5))
                test_acc.update(metrics[0].item(), inputs.size(0))

                # if idx % 200 == 0:
            print('STL-10 accuracy: %.3f %%' % (100 * correct / total))
            print("epoch:",epoch," test acc:",test_acc.avg)

            arr.append(test_acc.avg)
    arr=arr[2:]
    arr_mean = np.mean(arr)
    arr_var = np.var(arr)
    # 求总体标准差
    arr_std_1 = np.std(arr)
    # 求样本标准差
    arr_std_2 = np.std(arr, ddof=1)
    print("平均值为：%f" % arr_mean)
    print("方差为：%f" % arr_var)
    print("总体标准差为: %f" % arr_std_1)
    print("样本标准差为: %f" % arr_std_2)



if __name__ == '__main__':
    # transfer_resnet()

    tranfer_monilenet()
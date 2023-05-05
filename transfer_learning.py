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


# torchvision.datasets.STL10


import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the datasets
tinyimagenet_dataset = datasets.ImageFolder(root='../../data/tiny-imagenet-200/train', transform=transform)
stl10_dataset = datasets.STL10(root='../../data/', split='train', download=True, transform=transform)


# Load the pre-trained WRN-40-2 model and freeze its parameters
wrn40_2 = models.wide_resnet50_2(pretrained=True)
for param in wrn40_2.parameters():
    param.requires_grad = False

# Define the linear classifier to train on top of the last feature layer
num_classes_stl10 = 10
num_classes_tinyimagenet = 200
in_features = wrn40_2.fc.in_features
classifier_stl10 = nn.Linear(in_features, num_classes_stl10)
classifier_tinyimagenet = nn.Linear(in_features, num_classes_tinyimagenet)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_stl10 = torch.optim.SGD(classifier_stl10.parameters(), lr=0.1, momentum=0.9)
optimizer_tinyimagenet = torch.optim.SGD(classifier_tinyimagenet.parameters(), lr=0.1, momentum=0.9)

# Train the linear classifier on the STL-10 dataset
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(stl10_dataset, 0):
        inputs, labels = data
        features = wrn40_2.features(inputs)
        features = features.view(features.size(0), -1)
        outputs = classifier_stl10(features)
        loss = criterion(outputs, labels)
        optimizer_stl10.zero_grad()
        loss.backward()
        optimizer_stl10.step()
        running_loss += loss.item()
    print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / len(stl10_dataset)))

# Test the linear classifier on the STL-10 dataset
correct = 0
total = 0
with torch.no_grad():
    for data in stl10_dataset:
        images, labels = data
        features = wrn40_2.features(images)
        features = features.view(features.size(0), -1)
        outputs = classifier_stl10(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('STL-10 accuracy: %d %%' % (100 * correct / total))

# Train the linear classifier on the TinyImageNet dataset
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(tinyimagenet_dataset, 0):
        inputs, labels = data
        features = wrn40_2.features(inputs)
        features = features.view(features.size(0), -1)
        outputs = classifier_tinyimagenet(features)
        loss = criterion(outputs, labels)
        optimizer_tinyimagenet.zero_grad()
        loss.backward()
        optimizer_tinyimagenet.step()
        running_loss += loss.item()
    print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / len(tinyimagenet_dataset)))

# Test the linear classifier on the TinyImageNet dataset
correct = 0
total = 0
with torch.no_grad():
    for data in tinyimagenet_dataset:
        images, labels = data
        features = wrn40_2.features(images)
        features = features.view(features.size(0), -1)
        outputs = classifier_tinyimagenet(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('TinyImageNet accuracy: %d %%' % (100 * correct / total))

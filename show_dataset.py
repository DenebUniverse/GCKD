import os

import torch
from torchvision.datasets import ImageNet, ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# add data transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

train_folder = os.path.join('/mnt/data/wangbo/imagenet/', 'train')


# train_folder = os.path.join('../../../wangbo/data/imagenet')


# train_set = ImageFolder(train_folder, transform=train_transform)

# 创建一个函数，它接收一个目标类的ID（这里是215），
# 并返回一个函数，该函数接收一个元组（image, target），
# 并仅返回当目标等于目标类ID时的元组
def get_target_transform(target_class_id):
    def target_transform(target):
        return target == target_class_id

    return target_transform


# 创建ImageNet数据集的实例，仅加载 Class 215
train_set = ImageFolder(root=train_folder,
                        # split='train',
                        transform=transforms.Compose([transforms.ToTensor()]),
                        target_transform=get_target_transform(215))

test_loader = DataLoader(train_set,
                         batch_size=1,
                         shuffle=True,
                         num_workers=1,
                         pin_memory=True)

save_folder = '/mnt/data/wangbo/save_augmented/'
# 创建数据集的索引列表，其中class_idx是215类的索引
cnt = 0
for i, (img, label) in enumerate(test_loader):
    label=label.item()
    print(label)
    if label:
        # Create the filename for the augmented image
        filename = f"{label}_{i}.jpg"
        # Save the image to disk
        torch.save(img[0], os.path.join(save_folder, filename))
        cnt+=1
    if cnt >= 25:
        print('end>>>>>>>>>>>>>>>>>>>>>>')
        break

# # 使用matplotlib显示图像
# fig, axs = plt.subplots(1, 5)
# for i, idx in enumerate(indices):
#     img, _ = train_set[idx]
#     axs[i].imshow(img.permute(1, 2, 0))
#     axs[i].axis('off')
# plt.show()

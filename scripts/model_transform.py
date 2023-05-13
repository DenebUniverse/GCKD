import torch

if __name__ == '__main__':
    state_dict = torch.load('./save/teachers/models/resnet34_imagenet_vanilla/resnet34-333f7ec4.pth')
    torch.save({
        # 'epoch': model['epoch'],
        'model': state_dict,
        # 'best_acc': model['best_acc1']
    }, './save/teachers/models/resnet34_imagenet_vanilla/resnet34_torchvision.pth')
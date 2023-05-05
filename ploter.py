import argparse

import matplotlib.pyplot as plt
import os
import pandas as pd

from distiller_zoo.GLKD import GCKD
from models import model_dict


def print_csv(files):
    for file in files:
        file_dir = "./save/student_tensorboards/" + file + "/log1.csv"
        df = pd.read_csv(file_dir)
        res = df[df.epoch != 'epoch'].astype('float')
        print(file)
        # print(df.shape)
        print(res[['train_acc', 'test_acc', 'test_acc_top5']].max())
        print(res[['train_loss', 'test_loss']].min())
        print('_______________________________________________')


def summary_csv(files):
    for file in files:
        file_dir = "./save/student_tensorboards/" + file + "/log1.csv"
        df = pd.read_csv(file_dir)
        res = df[df.epoch != 'epoch'].astype('float')
        print(file)
        # print(df.shape)
        print(res[['train_acc', 'test_acc', 'test_acc_top5']].max())
        print(res[['train_loss', 'test_loss']].min())
        print('_______________________________________________')


def load_csv2list(files, epochs, type='test_acc'):
    data_list = dict.fromkeys(files)
    for file in files:
        file_dir = "./save/students/tensorboard/" + file + "/log0.csv"
        df = pd.read_csv(file_dir)
        res = df[df.epoch != 'epoch'].astype('float')
        print(file)
        print(df.shape)
        print(res[['train_acc', 'test_acc', 'test_acc_top5']].max())
        print(res[['train_loss', 'test_loss']].min())
        data_list[file] = res[type][:epochs]
    return data_list


def plot_list(x, y, label, title, xdes, ydes, path, x_scale="linear", dpi=300):
    plt.style.use('fivethirtyeight')  # bmh, fivethirtyeight, Solarize_Light2
    plt.figure(figsize=(10, 8))
    colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:cyan',
              'tab:gray', 'tab:brown', 'tab:purple', 'tab:olive', 'tab:pink']
    assert len(x) == len(y)
    for i in range(len(x)):
        if i < len(label):
            plt.plot(x[i], y[i], color=colors[i], label=label[i], linewidth=1.5)  # linewidth=1.5
        else:
            plt.plot(x[i], y[i], color=colors[i % len(label)], linewidth=1.5)  # linewidth=1.5

    plt.gca().get_xaxis().get_major_formatter().set_scientific(False)
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel(xdes, fontsize=24)
    plt.ylabel(ydes, fontsize=24)

    plt.title(title, fontsize=24)
    # my_y_ticks = np.arange(0, 1.1, 0.2)
    # plt.yticks(my_y_ticks, fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.legend(loc='lower right', fontsize=16)
    plt.legend(fontsize=16)
    plt.xscale(x_scale)
    # plt.margins(x=0)

    # plt.grid(True)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close("all")
    pass


def plot_train_curve():
    epochs = 240
    epoch = [i for i in range(epochs)]
    files = [
        ############################
        'S_resnet8x4-T_resnet32x4-D_cifar100_64-M_srrl-G_GCN_None-L_None-c_0.0-d_1.0-m_1.0-b_3.0-lr_None-adv_None_0.1-knn_8-clT_0.07-kdT_4-715',
        'S_resnet8x4-T_resnet32x4-D_cifar100_64-M_gld-G_GCN_None-L_None-c_0.0-d_1.0-m_1.0-b_3.0-lr_None-adv_None_0.1-knn_8-clT_0.07-kdT_4-767',

    ]
    out_dir = './save/students/plots/'
    # data_paths=["./save/student_tensorboards/"+file+"/log1.csv" for file in files]

    # label=[i for i in data_path]
    labels = [""
              ]
    acc_list = {"y": []}
    loss_list = {"y": []}
    save_path = ""

    for figure in ['train_acc', 'test_acc', 'test_acc_top5', 'train_loss', 'test_loss']:
        data_list = load_csv2list(files=files, type=figure, epochs=epochs)
        plot_list([epoch] * 2,
                  [data_list[file] for file in files],
                  label=[
                      "srrl",
                      "gld",
                  ],
                  title=figure, xdes="Epoch", ydes=figure,
                  path=os.path.join(out_dir, figure + "_srrl_gld.png")
                  )
    # summary_csv(files)


import re


def get_model_name(model_path, n_cls=None):
    """parse teacher name"""
    split_symbol = '_'
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '-T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1] + '_' + str(n_cls) if n_cls != None else name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    return segments[0] + '_' + str(n_cls) if n_cls != None else segments[0]


# get_student_name(student_path)

def plot_corr():
    import torch
    import torchvision
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import io
    import re

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-100 dataset
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    cifar100_train = torchvision.datasets.CIFAR100(root='../../data/cifar-100-python', train=True, download=True,
                                                   transform=transform)
    train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=int(len(cifar100_train) / 10), shuffle=False)

    n_cls = 100
    teacher_path = './save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth'
    # student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_gckd_2-G_TAG_1_one-A_64-adv_None_0.1_0.0-L_None-c_1.0-d_1.0-m_1.0-b_3.0-lr_None-clT_0.07-kdT_4-46_1/resnet8x4_best.pth'
    student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_gckd_2-G_TAG_1_one-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_1.0-m_1.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-160_1/resnet8x4_best.pth'
    match = re.search(r"gckd_(\d+)-G", student_path)
    if match:
        last_feature = match.group(1)
        print(last_feature)
    else:
        print("No match found.")

    # trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
    # trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)


    model_t = get_model_name(teacher_path, n_cls)
    model_s = get_model_name(student_path, n_cls)

    # Load model
    model_teacher = model_dict[model_t](num_classes=n_cls).to(device)
    model_teacher.load_state_dict(torch.load(teacher_path)['model'])
    cls_t = model_teacher.get_feat_modules()[-1]

    # Load the model state dictionary from a file into a buffer
    with open(student_path, 'rb') as f:
        buffer = io.BytesIO(f.read())

    checkpoint = torch.load(buffer)
    model_student = model_dict[model_s](num_classes=n_cls).to(device)
    model_student.load_state_dict(checkpoint['model'])


    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'imagenet':
        data = torch.randn(2, 3, 224, 224)
    elif opt.dataset == 'tinyimagenet':
        data = torch.randn(2, 3, 32, 32)

    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    s_n = feat_s[-1].shape[1] if last_feature == 1 else feat_s[-2].shape[1]
    t_n = feat_t[-1].shape[1] if last_feature == 1 else feat_t[-2].shape[1]
    parser = argparse.ArgumentParser('argument for training')
    opt=parse_option()
    criterion_kd = GCKD(s_dim=s_n, t_dim=t_n, m=0,opt=opt)  # m momentum rate one:0 two:1 momentum 0.99

    # Get class logits
    with torch.no_grad():
        images, _ = next(iter(train_loader))
        images = images.to(device)

        feat_t, logit_t = model_teacher(images, is_feat=True)
        feat_s, logit_s = model_student(images, is_feat=True)
        trans_feat_s, trans_feat_t, pred_feat_s = transfer(feat_s[-2], feat_t[-2], cls_t)
        pred_feat_s = cls_t(feat_s[-1])

        # logit_t=torch.softmax(logit_t, dim=1)
        # pred_feat_s = torch.softmax(pred_feat_s, dim=1)
        # feature_t=feat_t[-1].squeeze()
        # print(feature_t.s)

        corr_matrix_t = torch.einsum('ij,ik->ijk', logit_t, logit_t)
        corr_matrix_t = torch.sum(corr_matrix_t, dim=0)/logit_t.shape[0]
        corr_matrix_s = torch.einsum('ij,ik->ijk', pred_feat_s, pred_feat_s)
        corr_matrix_s = torch.sum(corr_matrix_s, dim=0)/pred_feat_s.shape[0]


        # corr_matrix_s = torch.mm(pred_feat_s.t(), pred_feat_s) / pred_feat_s.shape[0]
        # corr_matrix=torch.mm(logit_t.t(), pred_feat_s) / logit_t.shape[0]

        diff_corr_matrix=corr_matrix_t-corr_matrix_s


    # sns.heatmap(corr_matrix.cpu().numpy(), cmap='coolwarm')
    # plt.savefig('./save/plots/heatmap_l-T_{}-S_{}-sim.png'.format(get_model_name(teacher_path), get_model_name(student_path)))


    print(diff_corr_matrix.shape)
    # # Plot heatmap
    # sns.heatmap(diff_corr_matrix.cpu().numpy(), cmap='coolwarm')
    # # Save heatmap to file
    # plt.savefig('./save/plots/heatmap_l-T_{}-S_{}-dif.png'.format(get_model_name(teacher_path),get_model_name(student_path)))


    # # Plot heatmap
    # sns.heatmap(corr_matrix_t.cpu().numpy(), cmap='coolwarm')
    # # Save heatmap to file
    # plt.savefig(
    #     './save/plots/heatmap_l-T_{}-T_{}.png'.format(get_model_name(teacher_path), get_model_name(teacher_path)))

    # Plot heatmap
    sns.heatmap(corr_matrix_s.cpu().numpy(), cmap='coolwarm')
    # Save heatmap to file
    plt.savefig(
        './save/plots/heatmap_l-S_{}-S_{}.png'.format(get_model_name(student_path), get_model_name(student_path)))

plot_corr()

import argparse

import matplotlib.pyplot as plt
import os
import pandas as pd
from torch import nn
import torch.nn.functional as F

from distiller_zoo.GCKD import GCKD, SRRL, SimKD
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
        return segments[0] + '_' + segments[1] + '_' + segments[2] + '_' + str(n_cls) \
            if n_cls != None else segments[0] + '_' + segments[1] + '_' + segments[2]
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
    train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=int(len(cifar100_train) / 10), shuffle=True)

    n_cls = 100
    teacher_path = './save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth'
    # student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_gckd_2-G_TAG_1_one-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_1.0-m_1.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-160_1/resnet8x4_best.pth'
    # GCKD
    # student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_gckd_1-G_TAG_2_momentum-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_1.0-m_1.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-28_1/resnet8x4_best.pth'
    student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_gckd_1-G_TAG_2_momentum-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_0.0-m_0.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-617_1/resnet8x4_best.pth'
    # crd
    # student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_crd_1-G_TAG_2_momentum-A_8-adv_None_0.0_0.0-L_None-c_1.0-d_1.0-m_0.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-557_1/resnet8x4_best.pth'
    # kd
    # student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_kd_2-G_TAG_1_one-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_1.0-m_1.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-1500_1/resnet8x4_best.pth'
    # student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_kd_2-G_TAG_1_one-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_1.0-m_1.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-1500_1/resnet8x4_best.pth'
    # cls
    student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_kd_1-G_TAG_2_momentum-A_8-Q_4096-adv_None_0.1_0.0-L_None-c_1.0-d_0.0-m_0.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-208_1/resnet8x4_best.pth'
    #

    # # wrn
    # teacher_path = './save/teachers/models_240/wrn_40_2_vanilla/ckpt_epoch_240.pth'
    # # GCKD
    # student_path = './save/students/models/S_wrn_40_1-T_wrn_40_2-D_cifar100_64-M_gckd_1-G_TAG_2_momentum-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_1.0-m_1.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-1756_1/wrn_40_1_best.pth'
    # # kd
    # # student_path = './save/students/models/S_wrn_40_1-T_wrn_40_2-D_cifar100_64-M_kd_1-G_TAG_2_momentum-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_1.0-m_0.0-b_0.0-r_0.0-lr_None-clT_0.07-kdT_4-1618_1/wrn_40_1_best.pth'
    # # cls
    # student_path = './save/students/models/S_wrn_40_1-T_wrn_40_2-D_cifar100_64-M_kd_1-G_TAG_2_momentum-A_8-adv_None_0.0_0.0-L_None-c_1.0-d_0.0-m_0.0-b_0.0-r_0.0-lr_None-clT_0.07-kdT_4-1022_1/wrn_40_1_best.pth'
    distillation = 'cls'

    # params
    last_feature = 1
    # match = re.search(r"gckd_(\d+)-G", student_path)
    # if match:
    #     last_feature = match.group(1)
    #     print(last_feature)
    # else:
    #     print("No match found.")

    model_t = get_model_name(teacher_path, n_cls)
    model_s = get_model_name(student_path, n_cls)

    model_dict[model_t]
    # Load model
    model_teacher = model_dict[model_t](num_classes=n_cls).to(device)
    model_teacher.load_state_dict(torch.load(teacher_path)['model'])
    cls_t = model_teacher.get_feat_modules()[-1]

    # Load the model state dictionary from a file into a buffer
    with open(student_path, 'rb') as f:
        buffer = io.BytesIO(f.read())

    # load student
    checkpoint = torch.load(buffer)
    model_student = model_dict[model_s](num_classes=n_cls).to(device)
    model_student.load_state_dict(checkpoint['model'])

    # load projector
    if distillation == 'gckd':
        # 'cifar100':
        data = torch.randn(2, 3, 32, 32).to(device)
        model_teacher.eval()
        model_student.eval()
        feat_t, _ = model_teacher(data, is_feat=True)
        feat_s, _ = model_student(data, is_feat=True)

        s_dim = feat_s[-1].shape[1] if last_feature == 1 else feat_s[-2].shape[1]
        t_dim = feat_t[-1].shape[1] if last_feature == 1 else feat_t[-2].shape[1]

        transfer = SRRL(s_n=s_dim, t_n=t_dim).to(device) \
            if last_feature == 1 else SimKD(s_n=s_dim, t_n=t_dim, factor=2).to(device)
        transfer.load_state_dict(checkpoint['proj'][0])

    # Get class logits
    with torch.no_grad():
        images, _ = next(iter(train_loader))
        images = images.to(device)

        feat_t, logit_t = model_teacher(images, is_feat=True)
        if distillation == 'gckd':
            feat_s, _ = model_student(images, is_feat=True)
            # forward
            if last_feature == 1:
                trans_feat_s, pred_feat_s = transfer(feat_s[-1], cls_t)
            elif last_feature == 2:
                trans_feat_s, trans_feat_t, pred_feat_s = transfer(feat_s[-2], feat_t[-2], cls_t)

            pred_feat_s = cls_t(trans_feat_s)
        elif distillation in ['crd', 'kd', 'cls']:
            _, pred_feat_s = model_student(images, is_feat=True)

        logit_t = logit_t.div(torch.norm(logit_t, p=2, dim=0, keepdim=True))
        pred_feat_s = pred_feat_s.div(torch.norm(pred_feat_s, p=2, dim=0, keepdim=True))

        for method in ['lsum', 'lmm']:  # , 'psum', 'pmm']:
            if method == 'lsum':
                # # method1:lsum
                corr_matrix_t = torch.einsum('ij,ik->ijk', logit_t, logit_t)
                corr_matrix_t = torch.sum(corr_matrix_t, dim=0)  # / logit_t.shape[0]
                corr_matrix_s = torch.einsum('ij,ik->ijk', pred_feat_s, pred_feat_s)
                corr_matrix_s = torch.sum(corr_matrix_s, dim=0)  # / pred_feat_s.shape[0]
                # corr_matrix_ts= torch.einsum('ij,ik->ijk', logit_t, pred_feat_s)
                # corr_matrix_ts= torch.sum(corr_matrix_ts, dim=0) #/ pred_feat_s.shape[0]

                # corr_matrix_t = torch.einsum('ij,ik->ijk', logit_t, logit_t)/ logit_t.shape[0]
                # corr_matrix_s = torch.einsum('ij,ik->ijk', pred_feat_s, pred_feat_s)/ logit_t.shape[0]
                # corr_matrix_ts = torch.einsum('ij,ik->ijk', pred_feat_s, logit_t) / logit_t.shape[0]
                diff_corr_matrix = corr_matrix_t - corr_matrix_s
                # diff_corr_matrix = torch.sum(diff_corr_matrix, dim=0)

            elif method == 'lmm':
                # method2:lmm
                corr_matrix_t = torch.mm(logit_t.t(), logit_t)  # / logit_t.shape[0]
                corr_matrix_s = torch.mm(pred_feat_s.t(), pred_feat_s)  # / pred_feat_s.shape[0]
                # corr_matrix_ts = torch.mm(logit_t.t(), pred_feat_s) #/ logit_t.shape[0]
                diff_corr_matrix = corr_matrix_t - corr_matrix_s
            # elif method=='psum':
            #     # method3:psum
            #     p_t=F.softmax(logit_t /0.07, dim=1)
            #     p_s = F.softmax(pred_feat_s / 0.07, dim=1)
            #
            #     corr_matrix_t = torch.einsum('ij,ik->ijk', p_t, p_t)
            #     corr_matrix_t = torch.sum(corr_matrix_t, dim=0) / p_t.shape[0]
            #     corr_matrix_s = torch.einsum('ij,ik->ijk', p_s, p_s)
            #     corr_matrix_s = torch.sum(corr_matrix_s, dim=0) / p_s.shape[0]
            #     corr_matrix_ts = torch.einsum('ij,ik->ijk', p_t, p_s)
            #     corr_matrix_ts = torch.sum(corr_matrix_ts, dim=0) / p_s.shape[0]
            #     diff_corr_matrix = corr_matrix_t - corr_matrix_s
            # elif method=='pmm':
            #     # method4:pmm
            #     p_t=F.softmax(logit_t /0.07, dim=1)
            #     p_s = F.softmax(pred_feat_s / 0.07, dim=1)
            #
            #     corr_matrix_t = torch.mm(p_t.t(), p_t) / p_t.shape[0]
            #     corr_matrix_s = torch.mm(p_s.t(), p_s) / p_s.shape[0]
            #     corr_matrix_ts = torch.mm(p_t.t(), p_s) / p_t.shape[0]
            #     diff_corr_matrix = corr_matrix_t - corr_matrix_s

            out_path = './save/plot2/heatmap_' + distillation + '-' + method + '-'
            print(diff_corr_matrix.shape)
            # corr diff
            sns.heatmap(diff_corr_matrix.cpu().numpy(), center=0, cmap='PuOr', vmin=-0.4, vmax=0.4)
            plt.savefig(out_path + 'T_{}-S_{}.png'
                        .format(get_model_name(teacher_path), get_model_name(student_path)))
            plt.cla()
            plt.clf()

            # # corr teacher
            # sns.heatmap(corr_matrix_t.cpu().numpy(),center=0)#, cmap='coolwarm')
            # plt.savefig(out_path + 'T_{}-T_{}.png'
            #             .format(get_model_name(teacher_path), get_model_name(teacher_path)))
            # plt.cla()
            # plt.clf()
            #
            # # corr student
            # sns.heatmap(corr_matrix_s.cpu().numpy(),center=0)#, cmap='coolwarm')
            # plt.savefig(out_path + 'S_{}-S_{}.png'
            #             .format(get_model_name(student_path), get_model_name(student_path)))
            # plt.cla()
            # plt.clf()

            # # corr ts
            # sns.heatmap(corr_matrix_ts.cpu().numpy(),center=0)#, cmap='coolwarm')
            # plt.savefig(out_path + 'T_{}-S_{}.png'
            #             .format(get_model_name(teacher_path), get_model_name(student_path)))
            # plt.cla()
            # plt.clf()


# plot_corr()

def line_ploter():
    import seaborn as sns
    # sns.set_theme(style="darkgrid")
    sns.set(style="whitegrid", font_scale=2.4)

    # d = {'log2 k': [3,4,5,6,7,3, 4,5,6,7],
    #      'acc': [75.87, 75.84,75.99,76.44,62.82,
    #              76.65,76.4,75.93,76.15,62],
    #      'type':['share','share','share','share','share','momentum','momentum','momentum','momentum','momentum']}
    # df = pd.DataFrame(data=d)
    # # Plot the responses for different events and regions
    # plt.figure(dpi=300, figsize=(10, 8))
    # sns.lineplot(x="log2 k", y="acc",
    #              hue="type", style="type",
    #              data=df,
    #              linewidth=10,
    #              marker="s", markersize=20,
    #              # color='orange'
    #              )
    # plt.xticks([3, 4,5,6,7])

    # plt.figure(dpi=300, figsize=(10, 8))
    # d = {'node pert. rate': [0.0001,0.001,0.005,0.01, 0.1, 1, 10],
    #      'Test accuracy': [76.01,76.65,76.4,76.39, 76.31, 76.16, 76.2],
    #      }
    # df = pd.DataFrame(data=d)
    # g=sns.lineplot(data=df, x='node pert. rate', y='Test accuracy', linestyle='dashed',
    #                  linewidth=10,
    #                  marker="s", markersize=20, color='orange')
    # g.set(xlabel=None)
    # plt.xscale('log')
    # plt.ylim(76.0, 76.7)
    # plt.show()

    # plt.figure(dpi=300, figsize=(10, 8))
    # d = {'k of Adjacency Matrix': [4, 6, 8, 10, 12, 14, 16],
    #      'Test accuracy': [76.28, 76.18, 76.31, 76.12, 76.13, 76.14, 76.16],
    #      }
    # df = pd.DataFrame(data=d)
    # g = sns.lineplot(data=df, x='k of Adjacency Matrix', y='Test accuracy', linestyle='dashed',
    #                  linewidth=10,
    #                  marker="s", markersize=20, color='orange')
    # g.set(xlabel=None)
    # plt.xticks([4, 6, 8, 10,12,14,16])
    # plt.ylim(76.05, 76.35)
    # plt.tight_layout()
    # plt.show()

    # plt.figure(dpi=300, figsize=(10, 8))
    # d = {'depth of GNN': [1, 2, 3, 4,1, 2, 3, 4],
    #      'Test accuracy': [76.435, 76.23, 76.38, 76.26,76.58,76.45,76.19,75.89],
    #      'Params update type':['momentum','momentum','momentum','momentum','share','share','share','share']
    #      }
    # df = pd.DataFrame(data=d)
    # g=sns.lineplot(data=df, x='depth of GNN', y='Test accuracy',hue='Params update type', linestyle='dashed',
    #              linewidth=10,
    #              marker="s", markersize=20, color='orange'
    #                )
    # g.set(xlabel=None)
    # plt.ylim(75.8,76.6)
    # plt.xticks([1, 2, 3, 4])
    # # plt.show()

    # plt.figure(dpi=300, figsize=(10, 8))
    # d = {'depth of GNN': [1, 2, 3, 4,1, 2, 3, 4],
    #      'Test accuracy': [76.435, 76.23, 76.38, 76.26,76.58,76.45,76.19,75.89],
    #      'Params update type':['Top-1 accuracy','Top-1 accuracy','Top-1 accuracy','Top-1 accuracy',
    #                            'Top-5 accuracy','Top-5 accuracy','Top-5 accuracy','Top-5 accuracy']
    #      }
    # df = pd.DataFrame(data=d)
    # g=sns.lineplot(data=df, x='depth of GNN', y='Test accuracy',hue='Params update type', linestyle='dashed',
    #              linewidth=10,
    #              marker="s", markersize=20, color='orange'
    #                )
    # g.set(xlabel=None)
    # plt.ylim(75.8,76.6)
    # plt.xticks([1, 2, 3, 4])
    # # plt.show()

    plt.figure(dpi=300, figsize=(10, 8))
    df = pd.DataFrame({"depth of GNN": [1, 2, 3, 4],
                       "Top-1 accuracy": [76.435, 76.23, 76.38, 76.26],
                       "Top-5 accuracy": [94.41, 94.44, 94.37, 94.37]})
    g = sns.lineplot(data=df,x="depth of GNN", y="Top-1 accuracy", linestyle='dashed',
                 linewidth=10,
                 marker="s", markersize=20, color='orange',
                label="Top-1 acc"
                 )
    g.set_ylim(76.1,77.1)
    g2 = g.twinx()
    sns.lineplot(data=df,x="depth of GNN", y="Top-5 accuracy", ax=g2, linestyle='dashed',
            linewidth=10,
            marker="s", markersize=20,
                 label="Top-5 acc"
            )
    # g2.set_ylabel('Top-5 accuracy')
    # g.figure.legend()
    g2.set_ylim(94.1, 94.6)
    g.set(xlabel=None)
    g2.set(ylabel=None)
    # plt.legend(labels=['Top-1 acc', 'Top-5 acc'])
    # 获取第一个轴和第二个轴的线
    lines_1, labels_1 = g.get_legend_handles_labels()
    lines_2, labels_2 = g2.get_legend_handles_labels()
    # 合并图例
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    # 添加图例
    g.legend().set_visible(False)
    plt.legend(lines,labels)
    # g.set_ylabel('Test accuracy')
    g2.set_ylabel('Top-5 accuracy')
    # plt.ylim(75.8, 76.6)
    plt.xticks([1, 2, 3, 4])

    # plt.show()

    # plt.figure(dpi=300, figsize=(10, 8))
    # d = {'depth of GNN': [1, 2, 3, 4, 1, 2, 3, 4],
    #      'Top-5 accuracy': [94.41, 94.44, 94.37, 94.37, 94.34, 94.3, 94.02, 93.9],
    #      'Params update type': ['momentum', 'momentum', 'momentum', 'momentum', 'share', 'share', 'share', 'share']
    #      }
    # df = pd.DataFrame(data=d)
    # g = sns.lineplot(data=df, x='depth of GNN', y='Top-5 accuracy', hue='Params update type', linestyle='dashed',
    #                  linewidth=10,
    #                  marker="s", markersize=20, color='orange'
    #                  )
    # g.set(xlabel=None)
    # # plt.ylim(75.8, 76.6)
    # plt.xticks([1, 2, 3, 4])
    # plt.show()

    # plt.figure(dpi=300, figsize=(10, 8))
    # d = {'beta': [0.3, 3, 15, 30],
    #      'Test accuracy': [75.11, 76.38, 76.17, 75.6],
    #      }
    # df = pd.DataFrame(data=d)
    # g = sns.lineplot(data=df, x='beta', y='Test accuracy', linestyle='dashed',
    #                  linewidth=10,
    #                  marker="s",markersize=20, color='orange')
    # # plt.xlabel(r'$\beta$')
    # g.set(xlabel=None)
    # plt.xticks([0.3, 3,10, 15,20, 30])
    # # plt.xscale('log')
    # plt.ylim(75, 76.5)
    # plt.tight_layout()
    # plt.show()

    # df = pd.read_csv('./save/plot/losses.csv')
    # df=df[:150]
    # its = df[['Epoch', 'its']]
    # its=its.rename(columns={'its': 'Loss'})
    # its['type']=r'$\mathcal{L}_{\mathrm{IKD}}$'
    # gtt = df[['Epoch', 'gtt']]
    # gtt=gtt.rename(columns={'gtt': 'Loss'})
    # gtt['type'] = r'$\mathcal{L}_{\mathrm{GCL}}$'
    # gts = df[['Epoch', 'gts']]
    # gts=gts.rename(columns={'gts': 'Loss'})
    # gts['type'] =r'$\mathcal{L}_{\mathrm{GKD}}$'
    # data=pd.concat([its,gts,gtt])
    # data = data.reset_index()
    # plt.figure(dpi=300,figsize=(10,10))
    # g=sns.lineplot(data=data, x='Epoch',y='Loss',hue='type',
    #                # dashes=[(2, 2), (2, 2)],
    #                # linestyle='dashed',
    #                linestyle='--',
    #                linewidth=8,
    #                # marker=".",markersize=8, color='orange'
    #                )
    # # 获取当前图表的 Axes 对象
    # ax = plt.gca()
    # # 设置图例为虚线
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles, labels=labels)
    # for line in g.get_lines():
    #     line.set_linestyle('--')
    #     line.set_linewidth(6)
    # plt.legend()
    # g.legend_.set_title(None)

    # plt.show()

    # plt.figure(dpi=300, figsize=(10, 8))
    # d = {'Threshold': [-0.1,0,0.25,0.5,0.75],
    #      'Test accuracy': [76.77,76.78,76.35,76.45,76.28],
    #      }
    # df = pd.DataFrame(data=d)
    # g=sns.lineplot(data=df, x='Threshold', y='Test accuracy', linestyle='dashed',
    #                linewidth=10,
    #                marker="s",markersize=20, color='orange')
    # g.set(xlabel=None)
    # # g.set_ylabel('Test accuracy',fontsize=2)
    # # plt.xscale('log')
    # plt.xticks([-0.15,0,0.25,0.5,0.75])
    # plt.ylim(76.2, 76.9)
    plt.tight_layout()
    plt.show()


# line_ploter()


# def bar_ploter():
#     # 字体
#     TNR = {'fontname': 'Times New Roman'}
#     Fs = {'fontname': 'Fangsong'}
#
#     # 设置画布大小
#     plt.figure(figsize=(12, 6))
#     figure, axes = plt.subplots(1, 1, figsize=(12, 6), dpi=1000)
#     data = [76.39,76.31,76.16,76.2]
#     labels = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]
#
#     # 数据顺序反转
#     # data.reverse()
#     # 标签顺序反转
#     # labels.reverse()
#
#     N = 12
#     x = np.arange(N)
#
#     # 自定义每根柱子的颜色
#     colors = ["#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D", "#59A14F", "#8CD17D", "#B6992D", "#F1CE63", "#499894",
#               "#86BCB6", "#E15759", "#E19D9A"]
#     # 颜色顺序反转
#     # colors.reverse()
#     # 绘制纵向柱形图
#     plt.bar(range(len(data)), data, tick_label=labels, color=colors)
#
#     # plt.barh(range(len(data)), data, tick_label=labels,color = colors)
#     # 添加大标题
#     plt.title("2021年各月份销售业绩(万元)", fontsize=20, **Fs)
#
#     # 给X轴定义标签
#     # plt.xlabel("月份",fontsize=15)
#
#     # 给Y轴定义标签
#     # plt.ylabel("销售额(万元)",fontsize=15)
#
#     # 依次给每根柱子添加数据标签,并把字体设置为新罗马体(教科书、论文的数字、公式一般都用新罗马体)
#     for i, j in zip(x, data):
#         plt.text(i, j + 0.05, '%.0f' % j, ha='center', va='bottom', fontsize=15, **TNR)
#
#     # 为了美观，不显示画布的黑色边框
#     [axes.spines[loc_axis].set_visible(False) for loc_axis in ['top', 'right', 'bottom', 'left']]
#
#     # 不显示Y轴坐标
#     axes.set_yticks([])
#
#     # # 输出为矢量图，不管放大或缩小，图形皆不会失真
#     # plt.savefig(r"C:\Users\Administrator\Desktop\test.svg", format="svg")
#     # # 输出为常规的png格式
#     # plt.savefig(r"C:\Users\Administrator\Desktop\test.png", format="png")
#     # # 输出为常规的jpg格式
#     # plt.savefig(r"C:\Users\Administrator\Desktop\test.jpg", format="png")
#
#     # 绘图
#     plt.show()

def umap_ploter():
    import umap
    import matplotlib.pyplot as plt
    import torch
    import torchvision
    import io
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    import gc

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-100 dataset
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    cifar100_train = torchvision.datasets.CIFAR100(root='../../data/cifar-100-python', train=True, download=True,
                                                   transform=transform)
    # train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=int(len(cifar100_train) / 10), shuffle=True)
    method = 'gckd'
    embedding_type = 'IL'
    # length=1000
    # # Load CIFAR-100 sub-dataset
    # subset = Subset(cifar100_train, range(1, length))
    # desired_class = 2  # 想要加载的类别
    # targets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # targets=[1, 5, 7, 8, 9,10,11,13,14,17]
    # targets = [1, 5, 7, 8, 9, 13, 14, 17,20,22]
    targets = [1, 5, 7, 9, 13, 14, 17, 20, 22, 24]
    # targets = [1, 5, 7, 9, 13, 14, 17, 20, 22, 25]
    mapping_dict = {v: k * 2 for k, v in enumerate(targets)}
    print(mapping_dict)

    desired_indices = [i for i, target in enumerate(cifar100_train.targets[:18000]) if
                       target in targets]
    subset = Subset(cifar100_train, desired_indices)
    # batch_size = 64
    train_loader = DataLoader(subset, batch_size=len(subset), shuffle=True)
    del desired_indices
    gc.collect()

    # load model
    # teacher
    teacher_path = './save/teachers/models/resnet32x4_vanilla/resnet32x4_best.pth'
    # student
    if method == 'gckd':
        # student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_gckd_1-G_TAG_2_momentum-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_1.0-m_1.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-28_1/resnet8x4_best.pth'
        student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_gckd_1-G_TAG_2_momentum-A_8-adv_None_0.1_0.0-L_None-c_1.0-d_0.0-m_1.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-261_1/resnet8x4_best.pth'
    elif method == 'crd':
        student_path = './save/students/models/S_resnet8x4-T_resnet32x4-D_cifar100_64-M_crd_1-G_TAG_2_momentum-A_8-adv_None_0.0_0.0-L_None-c_1.0-d_1.0-m_0.0-b_3.0-r_1.0-lr_None-clT_0.07-kdT_4-557_1/resnet8x4_best.pth'
    n_cls = 100

    model_t = get_model_name(teacher_path, n_cls)
    model_s = get_model_name(student_path, n_cls)

    model_teacher = model_dict[model_t](num_classes=n_cls).to(device)
    model_teacher.load_state_dict(torch.load(teacher_path)['model'])
    cls_t = model_teacher.get_feat_modules()[-1]

    # Load the model state dictionary from a file into a buffer
    with open(student_path, 'rb') as f:
        buffer = io.BytesIO(f.read())

    # load student
    checkpoint = torch.load(buffer)
    model_student = model_dict[model_s](num_classes=n_cls).to(device)
    model_student.load_state_dict(checkpoint['model'])

    # load gckd
    if method == 'gckd':
        last_feature = 1

        # load projector

        # 'cifar100':
        data = torch.randn(2, 3, 32, 32).to(device)
        model_teacher.eval()
        model_student.eval()
        feat_t, _ = model_teacher(data, is_feat=True)
        feat_s, _ = model_student(data, is_feat=True)

        s_dim = feat_s[-1].shape[1] if last_feature == 1 else feat_s[-2].shape[1]
        t_dim = feat_t[-1].shape[1] if last_feature == 1 else feat_t[-2].shape[1]

        transfer = SRRL(s_n=s_dim, t_n=t_dim).to(device) \
            if last_feature == 1 else SimKD(s_n=s_dim, t_n=t_dim, factor=2).to(device)
        transfer.load_state_dict(checkpoint['proj'][0])

    if embedding_type == 'IE':
        # get one batch

        images, labels = next(iter(train_loader))
        labels = torch.tensor([mapping_dict[x.item()] for x in labels])
        images = images.to(device)

        # forward
        feat_t, logit_t = model_teacher(images, is_feat=True)
        trans_feat_s, pred_feat_s = model_student(images, is_feat=True)

        del images
        gc.collect()
        if method == 'gckd':
            # projector
            if last_feature == 1:
                trans_feat_s, pred_feat_s = transfer(trans_feat_s[-1], cls_t)
            elif last_feature == 2:
                trans_feat_s, trans_feat_t, pred_feat_s = transfer(trans_feat_s[-2], feat_t[-2], cls_t)
        else:
            trans_feat_s = trans_feat_s[-1]
        npoit = feat_t[-1].shape[0]
        embedding_matrix_t = feat_t[-1].cpu().detach().numpy()
        embedding_matrix_s = trans_feat_s.cpu().detach().numpy()
        save_path = "./save/plot/UMAP_" + method + '_' + embedding_type
    elif embedding_type == 'IL':
        # get one batch

        images, labels = next(iter(train_loader))
        labels = torch.tensor([mapping_dict[x.item()] for x in labels])
        images = images.to(device)

        # forward
        feat_t, logit_t = model_teacher(images, is_feat=True)
        trans_feat_s, pred_feat_s = model_student(images, is_feat=True)

        del images
        gc.collect()
        if method == 'gckd':
            # projector
            if last_feature == 1:
                trans_feat_s, pred_feat_s = transfer(trans_feat_s[-1], cls_t)
            elif last_feature == 2:
                trans_feat_s, trans_feat_t, pred_feat_s = transfer(trans_feat_s[-2], feat_t[-2], cls_t)
        npoit = feat_t[-1].shape[0]
        embedding_matrix_t = logit_t.cpu().detach().numpy()
        embedding_matrix_s = pred_feat_s.cpu().detach().numpy()
        save_path = "./save/plot/UMAP_" + method + '_' + embedding_type

    # save
    # visual
    print(npoit)
    features = np.concatenate((embedding_matrix_t, embedding_matrix_s), axis=0)
    # UMAP
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(features)
    teacher_embeddings = embeddings[:npoit, :]
    student_embeddings = embeddings[npoit:, :]

    ## combine
    fig = plt.figure(figsize=(9, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=torch.cat((labels, labels + 1), 0), marker='o', cmap='tab20', s=2)
    plt.colorbar(boundaries=np.arange(len(targets) * 2 + 1) - 0.5).set_ticks(np.arange(len(targets) * 2))
    # plt.title('UMAP visualization embeddings')
    fig.tight_layout()
    plt.savefig(save_path + "ST.png")
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(student_embeddings[:, 0], student_embeddings[:, 1], c=labels, marker='o', cmap='tab10', s=2)
    # plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    # plt.title('UMAP visualization student embeddings')
    fig.tight_layout()
    plt.savefig(save_path + "S.png")
    plt.show()

    plt.cla()
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(teacher_embeddings[:, 0], teacher_embeddings[:, 1], c=labels, marker='o', cmap='tab10', s=2)
    # plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    # plt.title('UMAP visualization teacher embeddings')
    fig.tight_layout()
    plt.savefig(save_path + "T.png")
    plt.show()

# umap_ploter()

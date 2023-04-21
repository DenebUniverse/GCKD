import matplotlib.pyplot as plt
import os
import pandas as pd

epochs = 240


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


def load_csv2list(files, type='test_acc'):
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
    data_list = load_csv2list(files=files, type=figure)
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

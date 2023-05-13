import os
import pandas as pd

summary = pd.DataFrame()

in_dirs = "./save/students/tensorboard/"
files = os.listdir(in_dirs)
for file in files:
    if os.path.isfile(os.path.join(in_dirs, file)):
        files.remove(file)
        continue
    print(file)
    params = file.replace('-', '_').split("_")
    if len(params) == 37:
        res_dict = {"teacher_model": params[3], "student_model": params[1],
                    "dataset": params[5], "batch_size": params[6],
                    "distill": params[8], "GNN": params[10], "layers": params[11], "encoders": params[12],
                    "Adjacency": params[14], "NPerturb": params[17], "Eperturb": params[18],
                    "loss": params[20], "cls": params[22], "div": params[24], "mu": params[26], "kd": params[28],
                    "test_acc": 0, "test_acc_top5": 0, "train_acc": 0, "train_loss": 0, "test_loss": 0, }
    else:
        res_dict = {"teacher_model": params[3], "student_model": params[1],
                    "dataset": params[5], "batch_size": params[6],
                    "distill": params[8], "GNN": params[10], "layers": 1, "encoders": params[11],
                    "Adjacency": params[13], "NPerturb": params[16], "Eperturb": params[17],
                    "loss": params[19], "cls": params[21], "div": params[23], "mu": params[25], "kd": params[27],
                    "test_acc": 0, "test_acc_top5": 0, "train_acc": 0, "train_loss": 0, "test_loss": 0, }
    file_dir = "./save/students/tensorboard/" + file + "/log1.csv"
    df = pd.read_csv(file_dir)
    res = df[df.epoch != 'epoch'].astype('float')

    acc = res[['train_acc', 'test_acc', 'test_acc_top5', 'epoch']].max()
    loss = res[['train_loss', 'test_loss']].min()
    res_dict['epoch'] = acc['epoch']
    res_dict['test_acc'] = acc['test_acc']
    res_dict['test_acc_top5'] = acc['test_acc_top5']
    res_dict['train_acc'] = acc['train_acc']
    res_dict['train_loss'] = loss['train_loss']
    res_dict['test_loss'] = loss['test_loss']
    summary = summary.append(res_dict, ignore_index=True)
summary.to_csv(in_dirs+"summary00.csv")

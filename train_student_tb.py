"""
the general training framework
"""

from __future__ import print_function

import csv
import os
import re
import argparse
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger
from torch.utils.tensorboard import SummaryWriter

from distiller_zoo.GLKD import GL_MoCo
from models import model_dict
from distiller_zoo.SimKD import ConvReg, SelfA, SRRL, SimKD

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader, get_dataloader_sample
# from dataset.imagenet_dali import get_dali_data_loader

from helper.loops import train_distill as train, validate_vanilla, validate_distill
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate

from crd.criterion import CRDLoss
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, VIDLoss, SemCKDLoss

split_symbol = '~' if os.name == 'nt' else ':'


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cos', action='store_true', default=None, help='use cosine lr schedule')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'], help='dataset')
    parser.add_argument('--model_s', type=str, default='resnet8x4')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--cl_T', type=float, default=0.07, help='temperature for CL distillation')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity', 'vid',
                                                                      'crd', 'semckd', 'srrl', 'simkd', 'gld'])
    parser.add_argument('-c', '--cls', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-d', '--div', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-m', '--mu', type=float, default=None, help='weight balance for feature l2 loss')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')
    parser.add_argument('-f', '--factor', type=int, default=2, help='factor size of SimKD')
    parser.add_argument('-s', '--soft', type=float, default=1.0, help='attention scale of SemCKD')

    # hint layer
    parser.add_argument('--hint_layer', default=1, type=int, choices=[0, 1, 2, 3, 4])

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # GCL
    # graph contrastive
    parser.add_argument('--gadv', default='None', type=str, choices=['adgcl', 'NPerturb', 'EPerturb', 'None'],
                        help='graph adverisal mode')
    parser.add_argument('--NPerturb', default=0.1, type=float)
    parser.add_argument('--EPerturb', default=0.1, type=float)
    parser.add_argument('--gnnlayer', default=None, type=str, choices=['GIN', 'GCN', 'TAG'])
    parser.add_argument('--gnnencoder', default=None, type=str, choices=['one', 'two', 'momentum'])
    parser.add_argument('--loss_func', default=None, type=str, choices=['softmax', 'cl'])
    parser.add_argument('--adj_k', type=int, default=20)

    # multiprocessing
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of teacher')

    opt = parser.parse_args()
    opt.skip_validation = True

    # set different learning rates for these MobileNet/ShuffleNet models
    if opt.model_s in ['MobileNetV2', 'MobileNetV2_1_0', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_1_5']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard
    opt.model_path = './save/students/models'
    opt.tb_path = './save/students/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    # set seed
    # if opt.deterministic:
    opt.seed = random.randint(1, 2048)

    def setup_seed(seed=1):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cudnn.deterministic = True
        cudnn.benchmark = False

    setup_seed(opt.seed)

    # model_name_template = split_symbol.join(['S', '{}_T', '{}_{}_{}_r', '{}_a', '{}_b', '{}_{}'])
    template = 'S_{}-T_{}-D_{}_{}-M_{}-G_{}_{}-A_{}-adv_{}_{}_{}-L_{}-c_{}-d_{}-m_{}-b_{}-lr_{}-clT_{}-kdT_{}-{}_{}'
    opt.model_name = template.format(opt.model_s, opt.model_t, opt.dataset, opt.batch_size, opt.distill,
                                                opt.gnnlayer, opt.gnnencoder, opt.adj_k,
                                                opt.gadv,opt.NPerturb,opt.EPerturb,
                                                opt.loss_func,
                                                opt.cls, opt.div, opt.mu, opt.beta,
                                                opt.cos, opt.cl_T, opt.kd_T, opt.seed, opt.trial,
                                                # opt.cls, opt.div, opt.beta, opt.trial
                                                )
    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    return segments[0]


def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    print('==> done')
    return model


best_acc = 0
total_time = time.time()


def main():
    opt = parse_option()

    # tensorboard logger
    print("tensorboard --logdir " + opt.tb_folder + "/tb_logs")
    opt.tb_writer = SummaryWriter(log_dir=opt.tb_folder + '/tb_logs', comment='GLD')
    with open(os.path.join(opt.tb_path, 'tensorbroad.txt'), 'a+') as f:
        f.write("tensorboard --logdir " + opt.tb_folder + "/tb_logs\n")
    save_file = os.path.join(opt.tb_path, opt.model_name, 'log{trial}.csv'.format(trial=opt.trial))
    with open(save_file, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_acc", "train_loss", "test_acc", "test_acc_top5", "test_loss",
             "losses_cls", "losses_kl", "losses_gtt", "losses_its", "losses_gts", "losses_mse",
             ])

    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    # ===================model=====================
    n_cls = {
        'cifar100': 100,
        'imagenet': 1000,
    }.get(opt.dataset, None)

    model_t = load_teacher(opt.path_t, n_cls, opt.gpu, opt)
    try:
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    except KeyError:
        print("This model is not supported.")

    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'imagenet':
        data = torch.randn(2, 3, 224, 224)

    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    # ===================criterion=====================
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    criterion_mse = nn.MSELoss()
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        if opt.dataset == 'cifar100':
            opt.n_data = 50000
        else:
            opt.n_data = 1281167
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'semckd':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = SemCKDLoss()
        self_attention = SelfA(opt.batch_size, s_n, t_n, opt.soft)
        module_list.append(self_attention)
        trainable_list.append(self_attention)
    elif opt.distill == 'srrl':
        s_n = feat_s[-1].shape[1]
        t_n = feat_t[-1].shape[1]
        model_fmsr = SRRL(s_n=s_n, t_n=t_n)
        criterion_kd = nn.MSELoss()
        module_list.append(model_fmsr)
        trainable_list.append(model_fmsr)
    elif opt.distill == 'simkd':
        s_n = feat_s[-2].shape[1]
        t_n = feat_t[-2].shape[1]
        model_simkd = SimKD(s_n=s_n, t_n=t_n, factor=opt.factor)
        criterion_kd = nn.MSELoss(opt)
        module_list.append(model_simkd)
        trainable_list.append(model_simkd)
    elif opt.distill == 'gld':
        s_n = feat_s[-1].shape[1]
        t_n = feat_t[-1].shape[1]
        # print("s_dim",s_n)
        # print("t_dim",t_n)
        momentum_rate={'one':0,'two':1,'momentum':0.99}
        criterion_kd = GL_MoCo(s_dim=s_n, t_dim=t_n,m=momentum_rate[opt.gnnencoder],opt=opt)#m momentum rate one:0 two:1 momentum 0.99
        module_list.append(criterion_kd.transfer)
        trainable_list.append(criterion_kd.transfer)
        # trainable_list.append(criterion_kd.gnn_q)
        # trainable_list.append(criterion_kd.gnn_k)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_mse)  # mse loss
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    module_list.append(model_t)

    # ===================optimizer=====================
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    optimizer_list=[optimizer]
    if opt.distill == 'gld':
        GNN_optimizer = optim.SGD(nn.ModuleList([criterion_kd.gnn_q, criterion_kd.gnn_k]).parameters(),
                                  lr=opt.learning_rate,
                                  momentum=opt.momentum,
                                  weight_decay=opt.weight_decay)
        optimizer_list.append(GNN_optimizer)
        if opt.gadv == 'adgcl':
            adversarial_optimizer = torch.optim.SGD(criterion_kd.mlp_edge_model.parameters(),
                                                    lr=0.001,  # opt.learning_rate,
                                                    momentum=opt.momentum,
                                                    weight_decay=opt.weight_decay)
            optimizer_list.append(adversarial_optimizer)


    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                module_list.cuda(opt.gpu)
                distributed_modules = []
                for module in module_list:
                    DDP = torch.nn.parallel.DistributedDataParallel
                    distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
                module_list = distributed_modules
                criterion_list.cuda(opt.gpu)
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion_list.cuda()
            module_list.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

    # ===================dataloader=====================
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                num_workers=opt.num_workers)
    elif opt.dataset == 'imagenet':
        if opt.dali is None:
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data, _, train_sampler = get_dataloader_sample(dataset=opt.dataset,
                                                                                           batch_size=opt.batch_size,
                                                                                           num_workers=opt.num_workers,
                                                                                           is_sample=True,
                                                                                           k=opt.nce_k,
                                                                                           multiprocessing_distributed=opt.multiprocessing_distributed)
            else:
                train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset=opt.dataset,
                                                                                  batch_size=opt.batch_size,
                                                                                  num_workers=opt.num_workers,
                                                                                  multiprocessing_distributed=opt.multiprocessing_distributed)
        else:
            # train_loader, val_loader = get_dali_data_loader(opt)
            pass
    else:
        raise NotImplementedError(opt.dataset)

    # ===================logger&validate=====================
    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if not opt.skip_validation:
        # validate teacher accuracy
        teacher_acc, _, _ = validate_vanilla(val_loader, model_t, criterion_cls, opt)

        if opt.dali is not None:
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print('teacher accuracy: ', teacher_acc)
    else:
        print('Skipping teacher validation.')

    # ===================routine=====================
    for epoch in range(1, opt.epochs + 1):
        torch.cuda.empty_cache()
        if opt.multiprocessing_distributed:
            if opt.dali is None:
                train_sampler.set_epoch(epoch)

        for the_optimizer in optimizer_list:
            adjust_learning_rate(epoch, opt, the_optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_acc_top5, train_loss, loss_dict = train(epoch, train_loader, module_list, criterion_list,
                                                                 optimizer_list, opt)
        time2 = time.time()

        opt.tb_writer.add_scalar('acc/train_acc', train_acc, epoch)
        opt.tb_writer.add_scalar('acc/train_acc_top5', train_acc_top5, epoch)
        for k, v in loss_dict.items():
            opt.tb_writer.add_scalar('loss/' + k, v, epoch)

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(epoch, opt.gpu, train_acc,
                                                                                        train_acc_top5, time2 - time1))

            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)

        print('GPU %d validating' % (opt.gpu))
        test_acc, test_acc_top5, test_loss = validate_distill(val_loader, module_list, criterion_cls, opt)

        opt.tb_writer.add_scalar('acc/test_acc', test_acc, epoch)
        opt.tb_writer.add_scalar('acc/test_acc_top5', test_acc_top5, epoch)
        for k, v in loss_dict.items():
            opt.tb_writer.add_scalar('loss/' + k, v, epoch)
        save_file = os.path.join(opt.tb_path, opt.model_name, 'log{trial}.csv'.format(trial=opt.trial))
        with open(save_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, round(train_acc, 5), round(train_loss, 5),
                 round(test_acc, 5), round(test_acc_top5, 5), round(test_loss, 5),
                 round(loss_dict['losses_cls'], 5),
                 round(loss_dict['losses_div'], 5),
                 round(loss_dict['losses_mse'], 5),
                 # round(loss_dict['losses_gtt'], 5),
                 # round(loss_dict['losses_its'], 5),
                 # round(loss_dict['losses_gts'], 5),
                 ])

        if opt.dali is not None:
            train_loader.reset()
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))

            logger.log_value('test_acc', test_acc, epoch)
            logger.log_value('test_loss', test_loss, epoch)
            logger.log_value('test_acc_top5', test_acc_top5, epoch)

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                }
                if opt.distill == 'simkd':
                    state['proj'] = trainable_list[-1].state_dict()
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))

                test_merics = {'test_loss': test_loss,
                               'test_acc': test_acc,
                               'test_acc_top5': test_acc_top5,
                               'epoch': epoch}

                save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
                print('saving the best model!')
                torch.save(state, save_file)

    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # This best accuracy is only for printing purpose.
        print('best accuracy:', best_acc)

        # save parameters
        save_state = {k: v for k, v in opt._get_kwargs() if k != 'tb_writer'}
        # No. parameters(M)
        num_params = (sum(p.numel() for p in model_s.parameters()) / 1000000.0)
        save_state['Total params'] = num_params
        save_state['Total time'] = (time.time() - total_time) / 3600.0
        params_json_path = os.path.join(opt.save_folder, "parameters.json")
        save_dict_to_json(save_state, params_json_path)

    opt.tb_writer.close()


if __name__ == '__main__':
    main()

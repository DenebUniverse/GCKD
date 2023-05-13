from __future__ import print_function, division
from cProfile import label

import sys
import time
import torch
from .util import AverageMeter, accuracy, reduce_tensor


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        if opt.dali is None:
            images, labels = batch_data
        else:
            images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        output = model(images)
        loss = criterion(output, labels)
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer_list, opt):
    """one epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    # ===================optimizer=====================
    optimizer = optimizer_list[0]
    if opt.distill in ['gld', 'gckd']:
        GNN_optimizer = optimizer_list[1]
        if opt.gadv == 'adgcl':
            adversarial_optimizer = optimizer_list[2]

    # ===================criterion=====================
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_mse = criterion_list[2]
    criterion_kd = criterion_list[3]
    # ===================model=====================
    model_s = module_list[0]
    model_t = module_list[-1]
    # ===================meter=====================
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_cls = AverageMeter()
    losses_div = AverageMeter()
    losses_kd = AverageMeter()
    if opt.distill in ['crd','srrl', 'simkd', 'gld', 'gckd']:
        losses_mse = AverageMeter()
        if opt.distill in ['gld', 'gckd']:
            losses_its = AverageMeter()
            losses_gts = AverageMeter()
            losses_gtt = AverageMeter()

    # ===================dataloader=====================
    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.dali is None:
            if opt.distill in ['crd']:
                images, labels, index, contrast_idx = data
            else:
                images, labels = data
        else:
            images, labels = data[0]['data'], data[0]['label'].squeeze().long()

        if opt.distill in ['semckd', 'gld', 'gckd'] and images.shape[0] < opt.batch_size:
            continue

        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        feat_s, logit_s = model_s(images, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else \
            model_t.get_feat_modules()[-1]
        # cls + kl div
        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd loss
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s, f_t = module_list[1](feat_s[opt.hint_layer], feat_t[opt.hint_layer])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'attention':
            # include 1, exclude -1.
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_mse=criterion_mse(f_s, f_t)
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'semckd':
            s_value, f_target, weight = module_list[1](feat_s[1:-1], feat_t[1:-1])
            loss_kd = criterion_kd(s_value, f_target, weight)
        elif opt.distill == 'srrl':
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
            loss_mse = criterion_mse(trans_feat_s, feat_t[-1])
            loss_kd = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, logit_t)
        elif opt.distill == 'simkd':
            trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            logit_s = pred_feat_s
            loss_kd = criterion_kd(trans_feat_s, trans_feat_t)
            loss_mse = criterion_mse(trans_feat_s, trans_feat_t)
        elif opt.distill == 'gld':
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
            loss_mse = criterion_mse(trans_feat_s, feat_t[-1])
            # loss_kd = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, logit_t)
            criterion_kd.use_forward_tt = False
            loss_kd, loss_its, loss_gts = criterion_kd(trans_feat_s, feat_t[-1])
            # loss_gtt= criterion_kd(trans_feat_s, feat_t[-1])
        elif opt.distill == 'gckd':
            if opt.last_feature == 1:
                trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
                loss_mse = criterion_mse(trans_feat_s, feat_t[-1])
                criterion_kd.use_forward_tt = False
                loss_kd, loss_its, loss_gts = criterion_kd(trans_feat_s, feat_t[-1])
            else:
                trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)
                loss_mse = criterion_mse(trans_feat_s, trans_feat_t)
                criterion_kd.use_forward_tt = False
                loss_kd, loss_its, loss_gts = criterion_kd(trans_feat_s, trans_feat_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.cls * loss_cls + opt.div * loss_div + opt.beta * loss_kd + opt.mu * loss_mse

        # ===================Metrics=====================
        losses.update(loss.item(), images.size(0))
        losses_cls.update(loss_cls.item(), images.size(0))
        losses_div.update(loss_div.item(), images.size(0))
        losses_kd.update(loss_kd.item(), images.size(0))
        if opt.distill in ['crd','srrl', 'simkd', 'gld', 'gckd']:
            losses_mse.update(loss_mse.item(), images.size(0))
            if opt.distill in ['gld', 'gckd']:
                losses_its.update(loss_its.item(), images.size(0))
                losses_gts.update(loss_gts.item(), images.size(0))
                # losses_gtt.update(loss_gtt.item(), images.size(0))
                pass
        metrics = accuracy(logit_s, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        # GNN_optimizer.zero_grad()
        optimizer.step()

        if opt.distill in ['gld', 'gckd']:
            # train GNN to minimize contrastive loss
            ## forward
            # if len(index)>=opt.knn:
            # freeze epoch
            if epoch == opt.lr_decay_epochs[0]:
                # 将模型参数设置为不需要梯度计算
                with torch.no_grad():
                    for param in criterion_kd.gnn_q.parameters():
                        param.requires_grad = False
                    for param in criterion_kd.gnn_k.parameters():
                        param.requires_grad = False
                criterion_kd.gnn_q.eval()
                criterion_kd.gnn_k.eval()
            if epoch < opt.lr_decay_epochs[0]:
                criterion_kd.use_forward_tt = True
                loss_contrast_gtt = criterion_kd(trans_feat_s, feat_t[-1])
                losses_gtt.update(loss_contrast_gtt.item(), images.size(0))
                ## backward
                GNN_optimizer.zero_grad()
                loss_contrast_gtt.backward()
                optimizer.zero_grad()
                GNN_optimizer.step()
                # train MPL to maximize contrastive loss
                if opt.gadv == 'adgcl':
                    ## forward
                    criterion_kd.use_forward_tt = True
                    loss_contrast_gtt = criterion_kd(trans_feat_s, feat_t[-1])
                    ## backward
                    adversarial_optimizer.zero_grad()
                    (-loss_contrast_gtt).backward()
                    adversarial_optimizer.step()

            # print info
            if idx % opt.print_freq == 0:
                print('Epoch: [{epoch}][{idx}/{batch_num}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Gpu {gpu} ({gpu})\n'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'losses_cls {losses_cls.val:.4f} ({losses_cls.avg:.4f})\t'
                      'losses_kl {losses_kl.val:.4f} ({losses_kl.avg:.4f})\t'
                      'losses_kd {losses_kd.val:.4f} ({losses_kd.avg:.4f})\t'
                      'losses_mse {losses_mse.val:.4f} ({losses_mse.avg:.4f})\n'
                      'losses_its {losses_its.val:.4f} ({losses_its.avg:.4f})\t'
                      'losses_gts {losses_gts.val:.4f} ({losses_gts.avg:.4f})\t'
                      'losses_gtt {losses_gtt.val:.4f} ({losses_gtt.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch=epoch, idx=idx, batch_num=len(train_loader), batch_time=batch_time, gpu=opt.gpu,
                    loss=losses, losses_cls=losses_cls, losses_kl=losses_div, losses_kd=losses_kd,
                    losses_mse=losses_mse,
                    losses_its=losses_its, losses_gts=losses_gts,
                    losses_gtt=losses_gtt,
                    top1=top1, top5=top5))
                sys.stdout.flush()
        else:
            # print info
            if idx % opt.print_freq == 0:
                print('Epoch: [{epoch}][{idx}/{batch_num}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Gpu {gpu} ({gpu})\n'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'losses_cls {losses_cls.val:.4f} ({losses_cls.avg:.4f})\t'
                      'losses_kl {losses_kl.val:.4f} ({losses_kl.avg:.4f})\t'
                      'losses_kd {losses_kd.val:.4f} ({losses_kd.avg:.4f})\t'
                      'losses_mse {losses_mse.val:.4f} ({losses_mse.avg:.4f})\n'
                # 'losses_its {losses_its.val:.4f} ({losses_its.avg:.4f})\t'
                # 'losses_gts {losses_gts.val:.4f} ({losses_gts.avg:.4f})\t'
                # 'losses_gtt {losses_gtt.val:.4f} ({losses_gtt.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch=epoch, idx=idx, batch_num=len(train_loader), batch_time=batch_time, gpu=opt.gpu,
                    loss=losses, losses_cls=losses_cls, losses_kl=losses_div, losses_kd=losses_kd,
                    losses_mse=losses_mse,
                    # losses_its=losses_its, losses_gts=losses_gts,
                    # losses_gtt=losses_gtt,
                    top1=top1, top5=top5))
                sys.stdout.flush()
    train_loss_dict = {"train_loss": losses.avg, "losses_cls": losses_cls.avg, "losses_div": losses_div.avg,
                       "losses_mse": losses_mse.avg, "losses_kd": losses_kd.avg,
                       # "losses_its": losses_its.avg, "losses_gts": losses_gts.avg,
                       }
    if opt.distill in ['gld', 'gckd']:
        train_loss_dict['losses_its'] = losses_its.avg
        train_loss_dict['losses_gts'] = losses_gts.avg
        train_loss_dict['losses_gtt'] = losses_gtt.avg
    return top1.avg, top5.avg, losses.avg, train_loss_dict


def validate_vanilla(val_loader, model, criterion, opt):
    """validation"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):

            if opt.dali is None:
                images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1)  # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    return top1.avg, top5.avg, losses.avg


def validate_distill(val_loader, module_list, criterion, opt):
    """validation"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    for module in module_list:
        module.eval()

    model_s = module_list[0]
    model_t = module_list[-1]
    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):

            if opt.dali is None:
                images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            if opt.distill == 'simkd':
                feat_s, _ = model_s(images, is_feat=True)
                feat_t, _ = model_t(images, is_feat=True)
                feat_t = [f.detach() for f in feat_t]
                cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else \
                    model_t.get_feat_modules()[-1]
                _, _, output = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            else:
                output = model_s(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1)  # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    return top1.avg, top5.avg, losses.avg

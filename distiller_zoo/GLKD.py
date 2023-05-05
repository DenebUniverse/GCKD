from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import TAGConv, GINConv, GraphConv
from dgl import DGLGraph
from scipy import sparse
# from dgl.nn.pytorch.factory import KNNGraph
import torch.nn.functional as F
import dgl.backend as B
# import dgl.function as fn
# import dgl
# import numpy as np

eps = 1e-7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def cos_distance_softmax(x):
    """
    soft:1*64*100
    w:1*64*1
    B.swapaxes():1*100*64
    """
    soft = F.softmax(x, dim=2)
    w = soft.norm(p=2, dim=2, keepdim=True)
    return 1 - soft @ B.swapaxes(soft, -1, -2) / (w @ B.swapaxes(w, -1, -2)).clamp(min=eps)


def knn_graph(x, k):
    """
    logit构建图
    x=logit:batch_size*100
    dist:1*batch_size*batch_size(对角线-1，其他值0-1)
    dst:1*batch_size*top_k(目标节点)
    """
    if B.ndim(x) == 2:
        x = B.unsqueeze(x, 0)
    n_samples, n_points, _ = B.shape(x)

    dist = cos_distance_softmax(x)

    fil = 1 - torch.eye(n_points, n_points)
    dist = dist * B.unsqueeze(fil, 0).cuda()
    dist = dist - B.unsqueeze(torch.eye(n_points, n_points), 0).cuda()

    k_indices = B.argtopk(dist, k, 2, descending=False)

    dst = B.copy_to(k_indices, B.cpu())
    src = B.zeros_like(dst) + B.reshape(B.arange(0, n_points), (1, -1, 1))

    per_sample_offset = B.reshape(B.arange(0, n_samples) * n_points, (-1, 1, 1))
    dst += per_sample_offset
    src += per_sample_offset
    dst = B.reshape(dst, (-1,))
    src = B.reshape(src, (-1,))
    adj = sparse.csr_matrix((B.asnumpy(B.zeros_like(dst) + 1), (B.asnumpy(dst), B.asnumpy(src))))

    g = DGLGraph(adj, readonly=True)
    # g = dgl.graph(adj)
    return g, {"adj": adj, "src": src, "dst": dst}


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, gnnlayer):
        super(Encoder, self).__init__()
        if gnnlayer == 'GIN':
            self.conv1 = GINConv(nn.Linear(in_dim, hidden_dim), 'max')  ###
        elif gnnlayer == 'GCN':
            self.conv1 = GraphConv(in_dim, hidden_dim, norm='both', weight=True, bias=True)
        elif gnnlayer == 'TAG':
            self.conv1 = TAGConv(in_dim, hidden_dim, k=1)
        self.l2norm = Normalize(2)

    def forward(self, g, eWeight=False):
        h = g.ndata['h']
        h = self.l2norm(self.conv1(g, h, edge_weight=g.edata['w'])) if eWeight else self.l2norm(self.conv1(g, h))
        return h


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, gnnlayer, num_layers):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        if gnnlayer == 'GIN':
            # input layer
            self.layers.append(GINConv(nn.Linear(in_dim, hidden_dim), 'max'))
            # hidden layers
            for i in range(num_layers - 1):
                self.layers.append(GINConv(nn.Linear(hidden_dim, hidden_dim), 'max'))
        elif gnnlayer == 'GCN':
            # input layer
            self.layers.append(GraphConv(in_dim, hidden_dim, norm='both', weight=True, bias=True))
            # hidden layers
            for i in range(num_layers - 1):
                self.layers.append(GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True))
        elif gnnlayer == 'TAG':
            # input layer
            self.layers.append(TAGConv(in_dim, hidden_dim, k=1))
            # hidden layers
            for i in range(num_layers - 1):
                self.layers.append(TAGConv(hidden_dim, hidden_dim, k=1))
        self.l2norm = Normalize(2)

    def forward(self, g, eWeight=False):
        h = g.ndata['h']
        for i, layer in enumerate(self.layers):
            if i > 0:
                h = F.relu(h)
            h = layer(g, h, edge_weight=g.edata['w']) if eWeight else layer(g, h)
        h = self.l2norm(h)
        return h

# transfer
class SRRL(nn.Module):
    """ICLR-2021: Knowledge Distillation via Softmax Regression Representation Learning"""

    def __init__(self, *, s_n, t_n):
        super(SRRL, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        ))

    def forward(self, feat_s, cls_t):
        feat_s = feat_s.unsqueeze(-1).unsqueeze(-1)
        temp_feat = self.transfer(feat_s)
        trans_feat_s = temp_feat.view(temp_feat.size(0), -1)

        pred_feat_s = cls_t(trans_feat_s)

        return trans_feat_s, pred_feat_s


class SimKD(nn.Module):
    """CVPR-2022: Knowledge Distillation with the Reused Teacher Classifier"""

    def __init__(self, *, s_n, t_n, factor=2):
        super(SimKD, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False,
                             groups=groups)

        # A bottleneck design to reduce extra parameters
        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n // factor),
            nn.BatchNorm2d(t_n // factor),
            nn.ReLU(inplace=True),
            conv3x3(t_n // factor, t_n // factor),
            # depthwise convolution
            # conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
            nn.BatchNorm2d(t_n // factor),
            nn.ReLU(inplace=True),
            conv1x1(t_n // factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        ))

    def forward(self, feat_s, feat_t, cls_t):

        # Spatial Dimension Alignment
        s_H, t_H = feat_s.shape[2], feat_t.shape[2]
        if s_H > t_H:
            source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
            target = feat_t
        else:
            source = feat_s
            target = F.adaptive_avg_pool2d(feat_t, (s_H, s_H))

        trans_feat_t = target

        # Channel Alignment
        trans_feat_s = getattr(self, 'transfer')(source)

        # Prediction via Teacher Classifier
        temp_feat = self.avg_pool(trans_feat_s)
        temp_feat = temp_feat.view(temp_feat.size(0), -1)
        pred_feat_s = cls_t(temp_feat)

        return self.avg_pool(trans_feat_s).squeeze(), self.avg_pool(trans_feat_t).squeeze(), pred_feat_s

# model

class GL_MoCo(nn.Module):
    def __init__(self, s_dim, t_dim=128, K=4096, m=0.99, T=0.1, bn_splits=8, symmetric=False, opt=None):
        super(GL_MoCo, self).__init__()

        self.gama=opt.gama
        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        self.t_dim = t_dim

        # # create the projection head

        self.transfer = SRRL(s_n=s_dim, t_n=t_dim)
        # self.transfer = SimKD(s_n=s_dim, t_n=t_dim)

        # self.gnn_q = Encoder(t_dim, t_dim, opt.gnnlayer)
        # self.gnn_k = Encoder(t_dim, t_dim, opt.gnnlayer)
        self.gnn_q = GNN(t_dim, t_dim, opt.gnnlayer, opt.layers)
        self.gnn_k = GNN(t_dim, t_dim, opt.gnnlayer, opt.layers)

        for param_t, param_s in zip(self.gnn_q.parameters(), self.gnn_k.parameters()):
            param_s.data.copy_(param_t.data)  # initialize
            param_s.requires_grad = False  # not update by gradient

        self.use_forward_tt = False
        self.adj_k = opt.adj_k
        self.graphadv = {'NPerturb': opt.NPerturb, "EPerturb": opt.EPerturb, "adv": opt.gadv}

        # create the queue
        self.register_buffer("queue", torch.randn(t_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.gnn_q.parameters(), self.gnn_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # print('ptr',ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def emb2graph(self, q, queue, adj_k=20, adj=None):  # adj share A
        X = torch.cat([q, queue.permute(1, 0)], dim=0)
        if adj == None:
            G, adj = knn_graph(X, k=adj_k)
        else:
            G = DGLGraph(adj, readonly=True)
        G = G.to(device)
        G.ndata['h'] = X
        return G, adj

    def contrastive_loss(self, im_q, im_k):
        batch_size = im_q.shape[0]
        # image contrastive loss
        # compute query features
        # q = self.encoder_t(im_q)  # queries: NxC
        q = im_q  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # # compute key features
        # with torch.no_grad():  # no gradient to keys
        #     # shuffle for making use of BN
        #     im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
        #
        #     k = self.encoder_s(im_k_)  # keys: NxC
        #     k = nn.functional.normalize(k, dim=1)  # already normalized
        #
        #     # undo shuffle
        #     k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
        k = im_k
        k = nn.functional.normalize(k, dim=1)  # already normalized

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss_its = nn.CrossEntropyLoss().cuda()(logits, labels)

        ## 构造图
        Graph_q, _ = self.emb2graph(q, self.queue.clone().detach(), adj_k=self.adj_k)
        Graph_k, _ = self.emb2graph(k, self.queue.clone().detach(), adj_k=self.adj_k)

        ## graph contrastive loss
        # compute query features
        q_g = self.gnn_q(Graph_q)[:batch_size]  # queries: NxC
        q_g = nn.functional.normalize(q_g, dim=1)  # already normalized

        k_g = self.gnn_k(Graph_k)[:batch_size]  # queries: NxC
        # # compute key features
        # with torch.no_grad():  # no gradient to keys
        #     # shuffle for making use of BN
        #     im_k_g, idx_unshuffle = self._batch_shuffle_single_gpu(k_g)
        #
        #     k_g = self.encoder_s(im_k_g)  # keys: NxC
        #     k_g = nn.functional.normalize(k_g, dim=1)  # already normalized
        #
        #     # undo shuffle
        #     k_g = self._batch_unshuffle_single_gpu(k_g, idx_unshuffle)
        k_g = nn.functional.normalize(k_g, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_g, k_g]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q_g, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss_gts = nn.CrossEntropyLoss().cuda()(logits, labels)

        # total loss
        loss = loss_its + loss_gts
        return loss, q, k, loss_its, loss_gts

    def contrastive_adv(self, im_q):
        batch_size = im_q.shape[0]
        # image contrastive loss
        # compute query features
        # q = self.encoder_t(im_q)  # queries: NxC
        q = im_q  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized
        k = q

        ## 构造图
        Graph_q, _ = self.emb2graph(q, self.queue.clone().detach(), adj_k=self.adj_k)
        Graph_k, adj = self.emb2graph(k, self.queue.clone().detach(), adj_k=self.adj_k)

        ## 图增强
        if self.graphadv['NPerturb'] > 0:
            Graph_k.ndata['h'] = Graph_k.ndata['h'] + torch.randn(Graph_k.num_nodes(), Graph_k.ndata['h'].shape[1]).to(
                device) * self.graphadv['NPerturb']
        if self.graphadv['EPerturb'] > 0:
            edge_logits = torch.rand((batch_size + self.K) * self.adj_k, 1).to(device)
            if self.graphadv['adv'] == 'adgcl':
                emb_src = Graph_k.ndata['h'][adj["src"]]
                emb_dst = Graph_k.ndata['h'][adj["dst"]]
                edge_emb = torch.cat([emb_src, emb_dst], 1)
                edge_logits = self.mlp_edge_model(edge_emb)
            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits * self.graphadv['EPerturb']) / temperature  # adv_eps[-1, 1]
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()
            Graph_k.edata['w'] = batch_aug_edge_weight

        ## graph contrastive loss
        # compute query features
        q_g = self.gnn_q(Graph_q)[:batch_size]  # queries: NxC
        q_g = nn.functional.normalize(q_g, dim=1)  # already normalized

        k_g = self.gnn_k(Graph_k)[:batch_size]  # queries: NxC
        k_g = nn.functional.normalize(k_g, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_g, k_g]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q_g, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss_gtt = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss_gtt

    def forward_st(self, ims, imt):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2, loss_its12, loss_gts12 = self.contrastive_loss(ims, imt)
            loss_21, q2, k1, loss_its21, loss_gts21 = self.contrastive_loss(ims, imt)
            loss = loss_12 + loss_21
            loss_its = loss_its12 + loss_its21
            loss_gts = loss_gts12 + loss_gts21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k, loss_its, loss_gts = self.contrastive_loss(ims, imt)

        self._dequeue_and_enqueue(k)

        return loss, loss_its, loss_gts

    def forward(self, ims, imt):
        return self.contrastive_adv(imt) if self.use_forward_tt else self.forward_st(ims, imt)


class GCKD(nn.Module):
    def __init__(self, s_dim, t_dim=128, K=4096, m=0.99, T=0.1, bn_splits=8, symmetric=False, opt=None):
        super(GCKD, self).__init__()

        self.gama=opt.gama
        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        self.t_dim = t_dim

        # # create the projection head

        self.transfer = SRRL(s_n=s_dim, t_n=t_dim) if opt.last_feature == 1 else SimKD(s_n=s_dim, t_n=t_dim,
                                                                                       factor=opt.factor)

        # self.gnn_q = Encoder(t_dim, t_dim, opt.gnnlayer)
        # self.gnn_k = Encoder(t_dim, t_dim, opt.gnnlayer)
        self.gnn_q = GNN(t_dim, t_dim, opt.gnnlayer, opt.layers)
        self.gnn_k = GNN(t_dim, t_dim, opt.gnnlayer, opt.layers)

        for param_t, param_s in zip(self.gnn_q.parameters(), self.gnn_k.parameters()):
            param_s.data.copy_(param_t.data)  # initialize
            param_s.requires_grad = False  # not update by gradient

        self.use_forward_tt = False
        self.adj_k = opt.adj_k
        self.graphadv = {'NPerturb': opt.NPerturb, "EPerturb": opt.EPerturb, "adv": opt.gadv}

        # create the queue
        self.register_buffer("queue", torch.randn(t_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.gnn_q.parameters(), self.gnn_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # print('ptr',ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def emb2graph(self, q, queue, adj_k=20, adj=None):  # adj share A
        X = torch.cat([q, queue.permute(1, 0)], dim=0)
        if adj == None:
            G, adj = knn_graph(X, k=adj_k)
        else:
            G = DGLGraph(adj, readonly=True)
        G = G.to(device)
        G.ndata['h'] = X
        return G, adj

    def contrastive_loss(self, im_q, im_k):
        batch_size = im_q.shape[0]
        # image contrastive loss
        # compute query features
        # q = self.encoder_t(im_q)  # queries: NxC
        q = im_q  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # # compute key features
        # with torch.no_grad():  # no gradient to keys
        #     # shuffle for making use of BN
        #     im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
        #
        #     k = self.encoder_s(im_k_)  # keys: NxC
        #     k = nn.functional.normalize(k, dim=1)  # already normalized
        #
        #     # undo shuffle
        #     k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
        k = im_k
        k = nn.functional.normalize(k, dim=1)  # already normalized

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss_its = nn.CrossEntropyLoss().cuda()(logits, labels)

        ## 构造图
        Graph_q, _ = self.emb2graph(q, self.queue.clone().detach(), adj_k=self.adj_k)
        Graph_k, _ = self.emb2graph(k, self.queue.clone().detach(), adj_k=self.adj_k)

        ## graph contrastive loss
        # compute query features
        q_g = self.gnn_q(Graph_q)[:batch_size]  # queries: NxC
        q_g = nn.functional.normalize(q_g, dim=1)  # already normalized

        # k_g = self.gnn_k(Graph_k)[:batch_size]  # queries: NxC
        embedding_g=self.gnn_k(Graph_k)  # queries: NxC
        k_g =embedding_g[:batch_size]
        queue_g=embedding_g[batch_size:]#
        # # compute key features
        # with torch.no_grad():  # no gradient to keys
        #     # shuffle for making use of BN
        #     im_k_g, idx_unshuffle = self._batch_shuffle_single_gpu(k_g)
        #
        #     k_g = self.encoder_s(im_k_g)  # keys: NxC
        #     k_g = nn.functional.normalize(k_g, dim=1)  # already normalized
        #
        #     # undo shuffle
        #     k_g = self._batch_unshuffle_single_gpu(k_g, idx_unshuffle)
        k_g = nn.functional.normalize(k_g, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_g, k_g]).unsqueeze(-1)
        # negative logits: NxK
        # l_neg = torch.einsum('nc,ck->nk', [q_g, self.queue.clone().detach()])
        l_neg = torch.einsum('nc,ck->nk', [q_g, queue_g.T])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss_gts = nn.CrossEntropyLoss().cuda()(logits, labels)

        # total loss
        loss = loss_its + loss_gts*self.gama
        return loss, q, k, loss_its, loss_gts

    def contrastive_adv(self, im_q):
        batch_size = im_q.shape[0]
        # image contrastive loss
        # compute query features
        # q = self.encoder_t(im_q)  # queries: NxC
        q = im_q  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized
        k = q

        ## 构造图
        Graph_q, _ = self.emb2graph(q, self.queue.clone().detach(), adj_k=self.adj_k)
        Graph_k, adj = self.emb2graph(k, self.queue.clone().detach(), adj_k=self.adj_k)

        ## 图增强
        if self.graphadv['NPerturb'] > 0:
            Graph_k.ndata['h'] = Graph_k.ndata['h'] + torch.randn(Graph_k.num_nodes(), Graph_k.ndata['h'].shape[1]).to(
                device) * self.graphadv['NPerturb']
        if self.graphadv['EPerturb'] > 0:
            edge_logits = torch.rand((batch_size + self.K) * self.adj_k, 1).to(device)
            if self.graphadv['adv'] == 'adgcl':
                emb_src = Graph_k.ndata['h'][adj["src"]]
                emb_dst = Graph_k.ndata['h'][adj["dst"]]
                edge_emb = torch.cat([emb_src, emb_dst], 1)
                edge_logits = self.mlp_edge_model(edge_emb)
            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits * self.graphadv['EPerturb']) / temperature  # adv_eps[-1, 1]
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()
            Graph_k.edata['w'] = batch_aug_edge_weight

        ## graph contrastive loss
        # compute query features
        q_g = self.gnn_q(Graph_q)[:batch_size]  # queries: NxC
        q_g = nn.functional.normalize(q_g, dim=1)  # already normalized

        embedding_g=self.gnn_k(Graph_k)  # queries: NxC
        k_g =embedding_g[:batch_size]
        queue_g=embedding_g[batch_size:]#
        k_g = nn.functional.normalize(k_g, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_g, k_g]).unsqueeze(-1)
        # negative logits: NxK
        # l_neg = torch.einsum('nc,ck->nk', [q_g, self.queue.clone().detach()])
        l_neg = torch.einsum('nc,ck->nk', [q_g, queue_g.T])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss_gtt = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss_gtt*self.gama

    def forward_st(self, ims, imt):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2, loss_its12, loss_gts12 = self.contrastive_loss(ims, imt)
            loss_21, q2, k1, loss_its21, loss_gts21 = self.contrastive_loss(ims, imt)
            loss = loss_12 + loss_21
            loss_its = loss_its12 + loss_its21
            loss_gts = loss_gts12 + loss_gts21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k, loss_its, loss_gts = self.contrastive_loss(ims, imt)

        self._dequeue_and_enqueue(k)

        return loss, loss_its, loss_gts

    def forward(self, ims, imt):
        return self.contrastive_adv(imt) if self.use_forward_tt else self.forward_st(ims, imt)

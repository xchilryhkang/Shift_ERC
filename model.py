import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_GAT import SemanticContextGraph
from shift import ShiftHead, build_shift_pairs, polarity_map


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if self.weight is None:
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


class BaselineModel(nn.Module):
    """
        h_i^m = Linear_m(x_i^m)                      project each modality to hidden_dim
        h_i^m = SemanticContextGraph({h^m})          optional (use_graph)
        h_i   = sum_m h_i^m
        y_i   = Linear(Dropout(ReLU(h_i)))           emotion classifier
        s_i   = ShiftHead(h, prev)                   optional (use_shift), [B, T, 2, 9]

    With use_graph = use_shift = False this is the plain no-context baseline.
    """

    def __init__(self, D_text, D_visual, D_audio, n_classes, hidden_dim, dropout, modals='tav',
                 dataset='IEMOCAP', use_graph=False, use_shift=False,
                 heads=4, layers=1, window=4, link_prev_same=False,
                 graph_dropout=0.1, attn_dropout=0.0, init_lambda=0.5, learn_prior=True, gate=True):
        super(BaselineModel, self).__init__()
        assert len(modals) > 0 and set(modals) <= set('tav'), "modals must be a subset of 'tav'"
        self.modals, self.use_graph, self.use_shift = modals, use_graph, use_shift

        in_dims = {'t': D_text, 'a': D_audio, 'v': D_visual}
        self.proj = nn.ModuleDict({m: nn.Linear(in_dims[m], hidden_dim) for m in modals})
        self.classifier = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))

        if use_graph:
            self.graph = SemanticContextGraph(hidden_dim, n_modals=len(modals), heads=heads, layers=layers,
                                              window=window, link_prev_same=link_prev_same,
                                              dropout=graph_dropout, attn_dropout=attn_dropout,
                                              init_lambda=init_lambda, learn_prior=learn_prior, gate=gate)
        if use_shift:
            self.shift = ShiftHead(hidden_dim, dropout)
            self.register_buffer('pol', polarity_map(dataset))

    def forward(self, textf, visuf, acouf, umask=None, qmask=None, lengths=None):
        # inputs: [T, B, D_m]  ->  outputs: [B, T, *]
        inputs = {'t': textf, 'a': acouf, 'v': visuf}
        hs = [self.proj[m](inputs[m].permute(1, 0, 2)) for m in self.modals]     # M x [B, T, H]
        if self.use_graph:
            hs = self.graph(hs, qmask, umask)
        h = sum(hs)                                                              # [B, T, H]

        logits = self.classifier(h)                                              # [B, T, C]
        shift_logits = None
        if self.use_shift:
            prev, _ = build_shift_pairs(qmask, umask)
            shift_logits = self.shift(h, prev)                                   # [B, T, 2, 9]
        return F.log_softmax(logits, dim=-1), F.softmax(logits, dim=-1), h, shift_logits

    def param_groups(self, lr, prior_lr, weight_decay):
        """Prior scalars (b_rel, theta) need a much larger lr and no weight decay."""
        named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        prior = [p for n, p in named if n.endswith(('b_rel', 'theta'))]
        rest = [p for n, p in named if not n.endswith(('b_rel', 'theta'))]
        groups = [{'params': rest, 'lr': lr, 'weight_decay': weight_decay}]
        if prior:
            groups.append({'params': prior, 'lr': prior_lr, 'weight_decay': 0.0})
        return groups

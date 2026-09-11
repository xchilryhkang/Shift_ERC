import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Minimal baseline (no context modeling, no graph, no fusion module):
        h_i^m = Linear_m(x_i^m)                 project each modality to hidden_dim
        h_i   = sum_{m in modals} h_i^m          sum embeddings
        y_i   = Linear(Dropout(ReLU(h_i)))       classifier head
    Every utterance is classified independently of its context.
    """

    def __init__(self, D_text, D_visual, D_audio, n_classes, hidden_dim, dropout, modals='tav'):
        super(BaselineModel, self).__init__()
        assert len(modals) > 0 and set(modals) <= set('tav'), "modals must be a subset of 'tav'"
        self.modals = modals
        in_dims = {'t': D_text, 'a': D_audio, 'v': D_visual}
        self.proj = nn.ModuleDict({m: nn.Linear(in_dims[m], hidden_dim) for m in modals})
        self.classifier = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))

    def forward(self, textf, visuf, acouf, umask=None, qmask=None, lengths=None):
        # inputs: [T, B, D_m]  ->  outputs: [B, T, *]
        inputs = {'t': textf, 'a': acouf, 'v': visuf}
        h = sum(self.proj[m](inputs[m].permute(1, 0, 2)) for m in self.modals)  # [B, T, H]
        logits = self.classifier(h)                                               # [B, T, C]
        return F.log_softmax(logits, dim=-1), F.softmax(logits, dim=-1), h

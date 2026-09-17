import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_GAT import SemanticContextGraph
from shift import ShiftHead, build_shift_pairs, polarity_map, shift_labels
from sheaf import EmotionalGATGraph, EmotionalSheafGraph, consistency, predict_shift


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
        s_i   = ShiftHead(h1, prev)                  optional (use_shift), [B, T, 2, 9]
        h2    = graph2({h^m}, segments from s_i)     optional (graph2 != 'none')
        h     = (1-a) LN(h1) + a LN(h2)              a learnable
        y_i   = Linear(Dropout(ReLU(h)))             emotion classifier

    Graph 2 runs on the projected features (not on h1), so the two branches are parallel;
    only its edge set depends on the shift predictions. With use_graph = use_shift = False
    and graph2 = 'none' this is the plain no-context baseline.
    """

    def __init__(self, D_text, D_visual, D_audio, n_classes, hidden_dim, dropout, modals='tav',
                 dataset='IEMOCAP', use_graph=False, use_shift=False,
                 heads=4, layers=1, window=4, link_prev_same=False,
                 graph_dropout=0.1, attn_dropout=0.0, init_lambda=0.5, learn_prior=True, gate=True,
                 graph2='none', sheaf_d=4, sheaf_layers=2, sheaf_map='diag', sheaf_step=1.0,
                 graph2_heads=4, graph2_layers=1, graph2_dropout=0.1, oracle_shift=False,
                 graph2_inter_modal=True, graph2_per_modal=False,
                 shift_depth=0, shift_emo_dim=None, shift_compare='full', shift_tau=None,
                 shift_mode='pair', soft_weight=False, init_mu=0.1,
                 graph2_bidir=False):
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
            self.shift = ShiftHead(hidden_dim, dropout, depth=shift_depth,
                                   emo_dim=shift_emo_dim, compare=shift_compare,
                                   mode=shift_mode)
            self.register_buffer('pol', polarity_map(dataset))

        self.graph2, self.oracle_shift, self.shift_tau = graph2, oracle_shift, shift_tau
        if graph2 != 'none':
            assert use_shift, "graph2 needs the shift head to build the partition (--use_shift)"
            if graph2 == 'sheaf':
                self.emo = EmotionalSheafGraph(hidden_dim, n_modals=len(modals), d=sheaf_d,
                                               layers=sheaf_layers, map_type=sheaf_map,
                                               dropout=graph2_dropout, step_size=sheaf_step,
                                               inter_modal=graph2_inter_modal,
                                               per_modal=graph2_per_modal, bidir=graph2_bidir)
            elif graph2 == 'gat':
                self.emo = EmotionalGATGraph(hidden_dim, n_modals=len(modals), heads=graph2_heads,
                                             layers=graph2_layers, dropout=graph2_dropout,
                                             inter_modal=graph2_inter_modal,
                                             per_modal=graph2_per_modal,
                                             soft_weight=soft_weight, init_mu=init_mu,
                                             bidir=graph2_bidir)
            else:
                raise ValueError(f'unknown graph2: {graph2}')
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.alpha = nn.Parameter(torch.zeros(1))       # sigmoid(0) = 0.5 at init

    def forward(self, textf, visuf, acouf, umask=None, qmask=None, lengths=None, warmup=False,
                labels=None):
        # inputs: [T, B, D_m]  ->  outputs: [B, T, *]
        inputs = {'t': textf, 'a': acouf, 'v': visuf}
        x = [self.proj[m](inputs[m].permute(1, 0, 2)) for m in self.modals]      # M x [B, T, H]
        hs = self.graph(x, qmask, umask) if self.use_graph else x
        h = sum(hs)                                                              # [B, T, H]

        shift_logits = pol_logits = None
        if self.use_shift:
            prev, valid = build_shift_pairs(qmask, umask)
            shift_logits, pol_logits = self.shift(h, prev)                       # [B, T, 2, 9]

        if self.graph2 != 'none' and not warmup:
            if self.oracle_shift:
                # ceiling experiment: build the partition from ground-truth polarities
                assert labels is not None, '--oracle_shift needs the labels'
                p = self.pol[labels.clamp(min=0)]
                sp = (torch.gather(p.unsqueeze(-1).expand(-1, -1, 2), 1, prev) !=
                      p.unsqueeze(-1)) & valid
            else:
                sp = predict_shift(shift_logits.detach(), tau=self.shift_tau)   # [B, T, 2] bool
            # the hard partition uses argmax (not differentiable anyway), but c^es must keep
            # its gradient so the main loss can reach the shift head
            cons = consistency(shift_logits) if getattr(self.emo, 'soft_weight', False) else None
            h2 = sum(self.emo(x, qmask, umask, sp, prev, valid, cons=cons))
            a = torch.sigmoid(self.alpha)
            h = (1 - a) * self.ln1(h) + a * self.ln2(h2)

        logits = self.classifier(h)                                              # [B, T, C]
        return F.log_softmax(logits, dim=-1), F.softmax(logits, dim=-1), h, shift_logits, pol_logits

    def param_groups(self, lr, prior_lr, weight_decay):
        """Prior scalars (b_rel, theta) need a much larger lr and no weight decay."""
        named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        is_prior = lambda n: n.endswith(('b_rel', 'theta', 'alpha', 'theta_mu'))
        prior = [p for n, p in named if is_prior(n)]      # scalars: need a much larger lr
        rest = [p for n, p in named if not is_prior(n)]
        groups = [{'params': rest, 'lr': lr, 'weight_decay': weight_decay}]
        if prior:
            groups.append({'params': prior, 'lr': prior_lr, 'weight_decay': 0.0})
        return groups

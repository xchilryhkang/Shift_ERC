"""
Emotion-shift detection on top of the semantic context graph.

Emotion labels are mapped to 3 polarities (0 = neg, 1 = neu, 2 = pos); a shift pair
(u_zeta -> u_i) is then labelled with  y = 3 * pol(zeta) + pol(i)  in {0..8}.

Pairs are adjacent and taken from two perspectives (as in ESDCM):
    p_u(i) = nearest j < i with s_j == s_i   (speaker-specific)
    q_u(i) = nearest j < i with s_j != s_i   (cross-speaker)
A pair is dropped when no such j exists.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG, NEU, POS = 0, 1, 2

# index = emotion label id, value = polarity. Order follows the dataloaders.
POLARITY = {
    # hap  sad  neu  ang  exc  fru
    'IEMOCAP': [POS, NEG, NEU, NEG, POS, NEG],
    # neu  sur  fea  sad  joy  dis  ang
    'MELD':    [NEU, POS, NEG, NEG, POS, NEG, NEG],
}


def polarity_map(dataset, device=None):
    return torch.tensor(POLARITY[dataset], dtype=torch.long, device=device)


def build_shift_pairs(qmask, umask):
    """
    Returns prev [B, T, 2] long, valid [B, T, 2] bool.
        prev[..., 0] = p_u(i) (same speaker), prev[..., 1] = q_u(i) (other speaker)
        invalid entries are set to 0 in `prev` and False in `valid`.
    """
    B, T, _ = qmask.shape
    spk = qmask.argmax(-1)                                            # [B, T]
    valid_u = umask.bool()
    t = torch.arange(T, device=qmask.device)

    past = (t[None, :, None] > t[None, None, :]) & valid_u[:, None, :]   # [B, T(i), T(j)] j < i
    same = (spk[:, :, None] == spk[:, None, :]) & past
    cross = (spk[:, :, None] != spk[:, None, :]) & past

    prev, valid = [], []
    for cand in (same, cross):
        idx = torch.where(cand, t[None, None, :], torch.full_like(cand, -1, dtype=torch.long)).max(-1).values
        valid.append((idx >= 0) & valid_u)
        prev.append(idx.clamp(min=0))
    return torch.stack(prev, -1), torch.stack(valid, -1)                 # [B, T, 2]


def shift_labels(labels, prev, valid, pol):
    """labels [B, T] -> y [B, T, 2] in {0..8}, -100 (ignore_index) where invalid."""
    p = pol[labels.clamp(min=0)]                                         # [B, T] polarity
    y = 3 * torch.gather(p.unsqueeze(-1).expand(-1, -1, 2), 1, prev) + p.unsqueeze(-1)
    return y.masked_fill(~valid, -100)


class ShiftHead(nn.Module):
    """
    node features -> 9-way shift logits for the two adjacent pairs of every utterance.

    depth = 0 : one linear layer on [h_zeta || h_i]                    (original)
    depth >= 1: project both endpoints into a shared emotion space,
                    z = ReLU(W_emo h)                        (repeated `depth` times)
                then classify a comparison vector. `compare` picks what goes in:
                    'cat'  -> [z_zeta || z_i]
                    'full' -> [z_zeta || z_i || z_i - z_zeta || z_zeta * z_i]
    A shift is a comparison, and a single linear layer on a concatenation represents a
    difference poorly, which is what `full` is for.
    """

    def __init__(self, hidden_dim, dropout=0.1, n_pol=3, depth=0, emo_dim=None,
                 compare='full', mode='pair'):
        """
        mode = 'pair'     : only the pairwise head (original).
        mode = 'polarity' : only a per-utterance polarity head. A shift is then
                            pol(zeta) != pol(i), which is transitive by construction:
                            the pairwise head can claim u1~u3, u3~u5 but u1!~u5, and blocks
                            built from such predictions contradict themselves.
        mode = 'both'     : both heads. The pairwise head is pulled towards the outer product
                            P_pol(zeta) (x) P_pol(i) by a consistency KL, so it keeps its view
                            of the relation between two utterances while staying transitive.
        """
        super().__init__()
        assert compare in ('cat', 'full') and mode in ('pair', 'polarity', 'both')
        self.n_pol, self.depth, self.compare, self.mode = n_pol, depth, compare, mode
        d = emo_dim or hidden_dim // 2
        if mode in ('polarity', 'both'):
            self.pol_fc = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, d), nn.ReLU(),
                                        nn.Linear(d, n_pol))
        if mode == 'polarity':
            return                                             # no pairwise branch at all

        if depth == 0:
            self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_dim, n_pol ** 2))
            return

        proj, dim = [], hidden_dim
        for _ in range(depth):
            proj += [nn.Dropout(dropout), nn.Linear(dim, d), nn.ReLU()]
            dim = d
        self.proj = nn.Sequential(*proj)                       # shared by both endpoints
        n_part = 2 if compare == 'cat' else 4
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_part * d, n_pol ** 2))

    @staticmethod
    def _gather_pairs(h, prev):
        """h [B, T, D] -> (source, target), each [B, T, 2, D]."""
        z_i = h.unsqueeze(2).expand(-1, -1, 2, -1)
        z_s = torch.gather(z_i, 1, prev.unsqueeze(-1).expand(-1, -1, -1, h.size(-1)))
        return z_s, z_i

    def pair_from_polarity(self, pol_logits, prev):
        """Outer product of the two endpoint polarities -> [B, T, 2, n_pol^2] log-probs."""
        lp = pol_logits.log_softmax(-1)                        # [B, T, n_pol]
        lp_s, lp_i = self._gather_pairs(lp, prev)              # [B, T, 2, n_pol]
        return (lp_s.unsqueeze(-1) + lp_i.unsqueeze(-2)).flatten(-2)   # log P(zeta) + log P(i)

    def forward(self, h, prev):
        """
        h [B, T, H], prev [B, T, 2]
        -> logits [B, T, 2, n_pol^2], pol_logits [B, T, n_pol] or None
        """
        pol_logits = self.pol_fc(h) if self.mode in ('polarity', 'both') else None
        if self.mode == 'polarity':
            return self.pair_from_polarity(pol_logits, prev), pol_logits

        x = self.proj(h) if self.depth > 0 else h
        z_s, z_i = self._gather_pairs(x, prev)
        if self.depth == 0 or self.compare == 'cat':
            feat = torch.cat([z_s, z_i], -1)
        else:
            feat = torch.cat([z_s, z_i, z_i - z_s, z_s * z_i], -1)
        return self.fc(feat), pol_logits

    def consistency_loss(self, logits, pol_logits, prev, valid):
        """KL( P_pair || P_pol(zeta) (x) P_pol(i) ) over the valid pairs."""
        tgt = self.pair_from_polarity(pol_logits, prev)        # log-probs [B, T, 2, n_pol^2]
        kl = F.kl_div(tgt, logits.log_softmax(-1), log_target=True, reduction='none').sum(-1)
        return kl[valid].mean() if valid.any() else logits.sum() * 0

    @staticmethod
    def polarity_loss(pol_logits, labels, pol, umask, weight=None):
        """Cross-entropy of the per-utterance polarity head."""
        y = pol[labels.clamp(min=0)].masked_fill(~umask.bool(), -100)
        return F.cross_entropy(pol_logits.reshape(-1, pol_logits.size(-1)), y.reshape(-1),
                               weight=weight, ignore_index=-100)

    @staticmethod
    def loss(logits, y, weight=None, focal_gamma=0.0):
        """9-way transition loss; focal_gamma=0 recovers ordinary cross-entropy."""
        logits = logits.reshape(-1, logits.size(-1))
        y = y.reshape(-1)
        valid = y != -100
        if not valid.any():
            return logits.sum() * 0

        ce = F.cross_entropy(logits[valid], y[valid], weight=weight, reduction='none')
        if focal_gamma > 0:
            pt = torch.exp(-ce)
            ce = (1.0 - pt).pow(focal_gamma) * ce
        return ce.mean()

    def consistency(self, logits):
        """P(no shift) = sum of the diagonal entries (m -> m). [B, T, 2]"""
        p = logits.softmax(-1).view(*logits.shape[:-1], self.n_pol, self.n_pol)
        return p.diagonal(dim1=-2, dim2=-1).sum(-1)

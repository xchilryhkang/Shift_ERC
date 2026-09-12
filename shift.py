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
    """node features -> 9-way shift logits for the two adjacent pairs of every utterance."""

    def __init__(self, hidden_dim, dropout=0.1, n_pol=3):
        super().__init__()
        self.n_pol = n_pol
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_dim, n_pol * n_pol))

    def forward(self, h, prev):
        """h [B, T, H], prev [B, T, 2] -> logits [B, T, 2, 9]"""
        H = h.size(-1)
        src = torch.gather(h.unsqueeze(2).expand(-1, -1, 2, -1), 1,
                           prev.unsqueeze(-1).expand(-1, -1, -1, H))     # [B, T, 2, H]
        return self.fc(torch.cat([src, h.unsqueeze(2).expand(-1, -1, 2, -1)], -1))

    @staticmethod
    def loss(logits, y):
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)

    def consistency(self, logits):
        """P(no shift) = sum of the diagonal entries (m -> m). [B, T, 2]"""
        p = logits.softmax(-1).view(*logits.shape[:-1], self.n_pol, self.n_pol)
        return p.diagonal(dim1=-2, dim2=-1).sum(-1)
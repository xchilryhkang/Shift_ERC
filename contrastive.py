"""
Supervised contrastive loss on the graph-1 embeddings, plus a cosine cut rule.

Idea
----
Instead of a 9-class ShiftHead (which overfits: train bin-F1 ~87 vs test ~57), supervise the
graph-1 output h so that utterances of the same emotion (or polarity) sit close together and
different ones sit apart. Segment boundaries are then read straight from h: a large cosine
distance between consecutive utterances of a chain is a shift. `smoothness.py` already showed
cosine distance alone reaches AUC 0.77 for shift on the untrained-for-this h, so shaping h with
a contrastive loss should make the cut rule reliable enough to replace the head.

Two pieces:
  sup_con_loss(h, labels, umask) : SupCon (Khosla et al. 2020). Positives = same label in the
                                   same dialogue; the target utterance is included, others are
                                   negatives. `by` chooses 3-polarity or full-emotion targets.
  cosine_shift(h, qmask, umask, prev, valid, tau)
                                 : shift on each pair (p_u(i)->i and q_u(i)->i) when
                                   1 - cos(h_zeta, h_i) > tau. Returns the same [B,T,2] bool that
                                   predict_shift returns, so the partition code is unchanged.
                                   tau can be a scalar or two values (one per channel), since the
                                   two chains have very different shift rates.
"""
import torch
import torch.nn.functional as F

from shift import build_shift_pairs, polarity_map  # noqa: F401  (kept for callers)


def sup_con_loss(h, labels, umask, pol=None, temperature=0.1):
    """
    h      : [B, T, H]      graph-1 output
    labels : [B, T]         emotion labels
    pol    : optional [n_emotions] map to polarity; if given, contrast by polarity instead
    Within each dialogue, positives share the (mapped) label. Padding is excluded.
    """
    B, T, H = h.shape
    dev = h.device
    z = F.normalize(h, dim=-1)
    y = labels if pol is None else pol[labels.clamp(min=0)]
    ok = umask.bool()

    total, n = h.new_zeros(()), 0
    for b in range(B):
        m = ok[b]
        if m.sum() < 2:
            continue
        zb, yb = z[b, m], y[b, m]                                   # [L, H], [L]
        sim = (zb @ zb.t()) / temperature                          # [L, L]
        L = zb.size(0)
        self_mask = torch.eye(L, dtype=torch.bool, device=dev)
        sim = sim.masked_fill(self_mask, float('-inf'))
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)      # log softmax over non-self
        pos = (yb[:, None] == yb[None, :]) & ~self_mask             # [L, L]
        pos_cnt = pos.sum(1)
        valid = pos_cnt > 0                                         # rows with at least one positive
        if valid.any():
            per_row = (logp.masked_fill(~pos, 0.0).sum(1)[valid] / pos_cnt[valid])
            total = total - per_row.sum()
            n += int(valid.sum())
    return total / max(n, 1)


def cosine_shift(h, qmask, umask, prev, valid, tau=0.5):
    """
    Shift on each adjacency pair when the cosine distance exceeds tau.
    Returns [B, T, 2] bool aligned with build_shift_pairs / predict_shift.
    tau: float, or (tau_p, tau_q) for the same-speaker and other-speaker channels.
    """
    z = F.normalize(h, dim=-1)                                      # [B, T, H]
    z_i = z.unsqueeze(2).expand(-1, -1, 2, -1)
    z_s = torch.gather(z_i, 1, prev.unsqueeze(-1).expand(-1, -1, -1, z.size(-1)))
    dist = 1.0 - (z_s * z_i).sum(-1)                                # [B, T, 2]
    if isinstance(tau, (tuple, list)):
        tau = torch.tensor(tau, device=h.device).view(1, 1, 2)
    return (dist > tau) & valid


def cosine_consistency(h, qmask, umask, prev, valid):
    """c^es analogue for the soft weight: cos similarity mapped to [0, 1]."""
    z = F.normalize(h, dim=-1)
    z_i = z.unsqueeze(2).expand(-1, -1, 2, -1)
    z_s = torch.gather(z_i, 1, prev.unsqueeze(-1).expand(-1, -1, -1, z.size(-1)))
    return ((z_s * z_i).sum(-1).clamp(-1, 1) + 1) / 2              # [B, T, 2] in [0,1]


def cosine_adjacent(h, umask, tau=0.5):
    """Adjacent temporal shift: 1 - cos(h_{i-1}, h_i) > tau. Returns [B, T] bool (col 0 = False)."""
    z = F.normalize(h, dim=-1)
    d = 1.0 - (z[:, 1:] * z[:, :-1]).sum(-1)                # [B, T-1]
    out = torch.zeros(h.size(0), h.size(1), dtype=torch.bool, device=h.device)
    out[:, 1:] = (d > tau) & umask[:, 1:].bool() & umask[:, :-1].bool()
    return out

"""
Emotional context graph: segment partition from predicted emotion shifts, propagated by a
sheaf neural network (Neural Sheaf Diffusion, Bodnar et al. 2022).

Partition
---------
Predicted shifts cut the dialogue into two chains of blocks:
  * speaker chain : for each speaker, cut between u_{p_u(i)} and u_i when a shift is predicted
  * time chain    : cut between u_{i-1} and u_i when a shift is predicted
Every utterance belongs to at most 2 blocks. The context edge set is their UNION: one edge
per ordered pair (j -> i), j < i, j in a block with i. A binary flag `in_both` marks pairs
that lie in both blocks.

Edges carry 4 relations, like the semantic graph: self / mod / same / cross.
No edge weights: the partition alone decides connectivity. The structural prior
(b_r - lambda*phi) stays in the semantic graph, and soft weights from c^es are not used here.

Sheaf propagation
-----------------
Features [B, N, H] are reshaped to stalks [B, N, d, f] with H = d*f. Each (node, edge) pair
gets a restriction map F_{v<|e} in R^{d x d} predicted from both endpoints (diagonal by
default). The sheaf Laplacian

    Delta[i,i] = sum_{e ~ i} F_{i<|e}^T F_{i<|e},   Delta[i,j] = -F_{i<|e}^T F_{j<|e}

equals delta^T delta, hence symmetric PSD: declaring edges past -> present is fine, but
information still flows both ways along an edge. One layer is an Euler step of X' = -Delta X:

    X <- X - tau * sigma( Delta_hat (I x W1) X W2 )

Self-loops are dropped for the sheaf: delta on a self-loop is F x - F x = 0, so they
contribute nothing. Isolated nodes are safe because the normalisation adds I to the degree.

Conventions match the rest of the repo:
    features : list of M tensors [B, T, H]
    qmask    : [B, T, n_speakers] one-hot,  umask : [B, T]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

REL_SELF, REL_MOD, REL_SAME, REL_CROSS = 0, 1, 2, 3
N_REL = 4


# ---------------------------------------------------------------------------
# Shift predictions -> partition
# ---------------------------------------------------------------------------
def predict_shift(shift_logits, n_pol=3):
    """[B, T, 2, n_pol^2] -> [B, T, 2] bool, True when start polarity != target polarity."""
    c = shift_logits.argmax(-1)
    return (c // n_pol) != (c % n_pol)


def block_ids(shift_pred, qmask, prev, valid):
    """
    Block index of each utterance in both chains.

    shift_pred : [B, T, 2] bool, channel 0 = pair (p_u(i) -> i), channel 1 = (q_u(i) -> i)
    prev, valid: [B, T, 2] from shift.build_shift_pairs

    Returns blk [B, T, 2] long. Two utterances share a speaker block iff same speaker and
    same blk[...,0]; they share a time block iff same blk[...,1].
    """
    B, T, S = qmask.shape
    t = torch.arange(T, device=prev.device)

    # speaker chain: cut at i if (p_u(i) -> i) is a shift, or i is that speaker's first turn.
    # The running count runs along each speaker's own turns only, otherwise other speakers'
    # cuts would split a block.
    cut_spk = shift_pred[..., 0] | ~valid[..., 0]                         # [B, T]
    per_spk = torch.cumsum(cut_spk.unsqueeze(-1).float() * qmask, dim=1)  # [B, T, S]
    spk_blk = (per_spk * qmask).sum(-1).long()

    # time chain: cut at i if the adjacent pair (i-1 -> i) is a shift. The adjacent pair is
    # whichever channel points at i-1 (channel 0 when the same speaker just spoke, else 1).
    is_adj = (prev == (t[None, :, None] - 1)) & valid                     # [B, T, 2]
    cut_time = (shift_pred & is_adj).any(-1) | ~is_adj.any(-1)
    time_blk = torch.cumsum(cut_time.long(), dim=1)

    return torch.stack([spk_blk, time_blk], dim=-1)


def build_segment_edges(qmask, umask, shift_pred, prev, valid, n_modals):
    """
    Returns rel [B, N, N] long (-1 = no edge) and in_both [B, N, N] bool.
    Node index n = m * T + t; row = target, column = source (same layout as semantic_GAT).
    """
    B, T = umask.shape
    dev = umask.device
    N = n_modals * T

    blk = block_ids(shift_pred, qmask, prev, valid)                       # [B, T, 2]
    spk = qmask.argmax(-1)
    ok = umask.bool()
    t = torch.arange(T, device=dev)

    past = (t[None, :, None] > t[None, None, :]) & ok[:, None, :] & ok[:, :, None]
    e_spk = (blk[..., 0:1] == blk[..., 0].unsqueeze(1)) & \
            (spk[:, :, None] == spk[:, None, :]) & past
    e_time = (blk[..., 1:2] == blk[..., 1].unsqueeze(1)) & past
    ctx, both_u = e_spk | e_time, e_spk & e_time                          # [B, T, T]

    # lift utterance-level masks to (utterance, modality) nodes
    n = torch.arange(N, device=dev)
    mod_idx, t_idx = n // T, n % T
    same_mod = mod_idx[:, None] == mod_idx[None, :]
    same_time = t_idx[:, None] == t_idx[None, :]
    ok_n = ok[:, t_idx]                                                   # [B, N]
    pair_ok = ok_n[:, :, None] & ok_n[:, None, :]
    spk_n = spk[:, t_idx]
    same_spk_n = spk_n[:, :, None] == spk_n[:, None, :]

    ctx_n = ctx[:, t_idx][:, :, t_idx] & same_mod[None]
    rel = torch.full((B, N, N), -1, dtype=torch.long, device=dev)
    rel[(~same_mod & same_time)[None] & pair_ok] = REL_MOD
    rel[ctx_n & same_spk_n] = REL_SAME
    rel[ctx_n & ~same_spk_n] = REL_CROSS
    rel[(same_mod & same_time)[None].expand(B, N, N)] = REL_SELF   # also on padding: keeps
    #                                                        every GAT row non-empty (no NaN)
    in_both = both_u[:, t_idx][:, :, t_idx] & same_mod[None] & (rel >= 0)
    return rel, in_both


def to_edge_list(rel, in_both, drop_self=True):
    """[B, N, N] -> index [2, E] (batch-offset, row 0 = source), rel [E], both [E]."""
    mask = rel >= 0
    if drop_self:
        mask = mask & (rel != REL_SELF)
    b, i, j = mask.nonzero(as_tuple=True)                                 # i target, j source
    N = rel.size(1)
    return torch.stack([j + b * N, i + b * N]), rel[b, i, j], in_both[b, i, j].float()


# ---------------------------------------------------------------------------
# Sheaf layer
# ---------------------------------------------------------------------------
class SheafLayer(nn.Module):
    """One Euler step of sheaf diffusion over an edge list."""

    def __init__(self, d, f, map_type='diag', n_rel=N_REL, rel_dim=8, hidden=64,
                 dropout=0.0, step_size=1.0):
        super().__init__()
        assert map_type in ('diag', 'general')
        self.d, self.f, self.map_type, self.step, self.dropout = d, f, map_type, step_size, dropout
        H, out = d * f, d if map_type == 'diag' else d * d
        self.rel_emb = nn.Embedding(n_rel, rel_dim)
        self.sheaf_learner = nn.Sequential(
            nn.Linear(2 * H + rel_dim + 1, hidden), nn.ReLU(), nn.Linear(hidden, out))
        self.W1 = nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d))   # mixes stalk coords
        self.W2 = nn.Linear(f, f, bias=False)                             # mixes channels
        nn.init.xavier_uniform_(self.W2.weight)

    def restriction_maps(self, h, edge_index, rel, both):
        """Returns F_dst, F_src, each [E, d] (diag) or [E, d, d] (general)."""
        src, dst = edge_index
        ctx = torch.cat([self.rel_emb(rel), both.unsqueeze(-1)], -1)
        m_dst = torch.tanh(self.sheaf_learner(torch.cat([h[dst], h[src], ctx], -1)))
        m_src = torch.tanh(self.sheaf_learner(torch.cat([h[src], h[dst], ctx], -1)))
        if self.map_type == 'general':
            m_dst = m_dst.view(-1, self.d, self.d)
            m_src = m_src.view(-1, self.d, self.d)
        return m_dst, m_src

    def forward(self, x, edge_index, rel, both, n_nodes):
        """x: [n_nodes, d, f] -> same shape."""
        src, dst = edge_index
        h = F.dropout(x.reshape(n_nodes, -1), self.dropout, self.training)
        F_dst, F_src = self.restriction_maps(h, edge_index, rel, both)

        y = self.W2(x)                                                    # X W2  (mixes channels)
        y = torch.einsum('pq,nqf->npf', self.W1, y)                       # (I x W1) X W2

        if self.map_type == 'diag':
            deg = torch.zeros(n_nodes, self.d, device=x.device).index_add_(0, dst, F_dst ** 2)
            L = (deg + 1.0).sqrt().unsqueeze(-1)                          # D^{1/2}, +I for stability
            z = y / L                                                     # D^{-1/2} y
            out = deg.unsqueeze(-1) * z
            out = out.index_add_(0, dst, -(F_dst * F_src).unsqueeze(-1) * z[src])
            out = out / L
        else:
            dd = torch.einsum('eqp,eqr->epr', F_dst, F_dst)               # F_dst^T F_dst
            deg = torch.zeros(n_nodes, self.d, self.d, device=x.device).index_add_(0, dst, dd)
            eye = torch.eye(self.d, device=x.device).expand_as(deg)
            L = torch.linalg.cholesky(deg + eye)                          # D + I = L L^T
            z = torch.linalg.solve_triangular(L.transpose(1, 2), y, upper=True)
            out = torch.einsum('npq,nqf->npf', deg, z)
            FtF = torch.einsum('eqp,eqr->epr', F_dst, F_src)              # F_dst^T F_src
            out = out.index_add_(0, dst, -torch.einsum('epq,eqf->epf', FtF, z[src]))
            out = torch.linalg.solve_triangular(L, out, upper=False)

        return x - self.step * F.elu(out)


class EmotionalSheafGraph(nn.Module):
    """features (list of M x [B, T, H]) -> list of M x [B, T, H]"""

    def __init__(self, hidden_dim, n_modals=3, d=4, layers=2, map_type='diag',
                 dropout=0.1, step_size=1.0):
        super().__init__()
        assert hidden_dim % d == 0, 'hidden_dim must be divisible by the stalk dimension d'
        self.n_modals, self.d, self.f = n_modals, d, hidden_dim // d
        self.layers = nn.ModuleList([
            SheafLayer(d, self.f, map_type, dropout=dropout, step_size=step_size)
            for _ in range(layers)])

    def forward(self, features, qmask, umask, shift_pred, prev, valid):
        B, T, H = features[0].shape
        N = self.n_modals * T
        rel, both = build_segment_edges(qmask, umask, shift_pred, prev, valid, self.n_modals)
        edge_index, e_rel, e_both = to_edge_list(rel, both, drop_self=True)

        x = torch.cat(features, dim=1).reshape(B * N, self.d, self.f)
        for layer in self.layers:
            x = layer(x, edge_index, e_rel, e_both, B * N)
        return list(x.reshape(B, N, H).split(T, dim=1))


# ---------------------------------------------------------------------------
# GAT on the same edges — ablation that isolates the partition from the sheaf
# ---------------------------------------------------------------------------
class EmotionalGATGraph(nn.Module):
    def __init__(self, hidden_dim, n_modals=3, heads=4, layers=1, dropout=0.1):
        super().__init__()
        from semantic_GAT import StructuralPriorGATLayer
        self.n_modals, self.dropout = n_modals, dropout
        self.layers = nn.ModuleList([
            StructuralPriorGATLayer(hidden_dim, hidden_dim, heads=heads, init_lambda=0.0,
                                    learn_prior=False) for _ in range(layers)])

    def forward(self, features, qmask, umask, shift_pred, prev, valid):
        B, T, H = features[0].shape
        rel, _ = build_segment_edges(qmask, umask, shift_pred, prev, valid, self.n_modals)
        rel = rel + 1                      # semantic_GAT: 0 = no edge, 1..4 = self/mod/same/cross
        phi = torch.zeros(rel.size(1), rel.size(2), device=rel.device)
        x = torch.cat(features, dim=1)
        for layer in self.layers:
            x = x + F.elu(F.dropout(layer(x, rel, phi), self.dropout, self.training))
        return list(x.split(T, dim=1))

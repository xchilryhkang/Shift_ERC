"""
Graph 2 as a hypergraph, with a virtual node per utterance.

Why a hypergraph here
---------------------
A block of the partition is a GROUP of utterances that share an emotion, not a set of
pairwise relations. A hyperedge says exactly that: one hyperedge per (block, chain, modality)
instead of k(k-1) directed edges. It is also orientation-free, so the "past -> present vs both
ways" question disappears, and an utterance at the start of a block is never isolated.

Virtual node
------------
Measured on IEMOCAP, direct inter-modal edges hurt graph 2 (removing them and giving each
modality its own weights took 69.94 -> 72.36). So instead of connecting t/a/v to each other,
each utterance gets a fourth, virtual node, and one hyperedge {t, a, v, virtual} per utterance
routes all cross-modal exchange through it. The virtual node is initialised as the mean of its
modalities by default, or can be supplied by the caller (for example, the semantic output of
graph 1).

Over-smoothing
--------------
Plain hypergraph convolution pools a whole hyperedge and hands the SAME vector back to every
member, so two utterances of one block receive identical messages no matter how different they
are -- attention pooling does not fix this, since it only changes what the shared vector is.
Two options counter it:
  loo=True  : leave-one-out. The message a node gets from a hyperedge excludes its own
              contribution, which makes it node-specific and matches "aggregate the
              neighbours" in an ordinary GNN. On by default.
  attn=True : the pooling weight of each member is learned instead of 1/|e|, so a block-mate
              that does not fit is down-weighted.

Propagation (one layer):
    gather :  S_e = sum_{v in e} a_ve W x_v ,  A_e = sum_{v in e} a_ve   (a_ve = 1 or attention)
    message:  m_{e->v} = (S_e - a_ve W x_v) / (A_e - a_ve)               (plain S_e/A_e if loo=False)
    scatter:  o_v = sum_{e ~ v} w_e m_{e->v} / sum_{e ~ v} w_e
    update :  x <- x + ELU(dropout(o))                                   (residual; VUEMO has none)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sheaf import block_ids, time_block_ids


# ---------------------------------------------------------------------------
# Hypergraph construction
# ---------------------------------------------------------------------------
def build_segment_hyperedges(qmask, umask, shift_pred, prev, valid, n_modals,
                             virtual=True, virtual_in_segment=True,
                             chain='both', shift_adj=None):
    """
    Node id (batch-offset): b * (M + V) * T + m * T + t, with m = M for the virtual node.

    Hyperedges
      segment : one per (batch, chain, block, modality); chains are the speaker chain and
                the time chain, so an utterance belongs to at most 2 segment hyperedges
                per modality
      modality: one per (batch, utterance) holding {t, a, v, virtual}; only when virtual=True,
                since without the virtual node this would be a direct cross-modal hyperedge

    Returns edge_index [2, nnz] (row 0 = node, row 1 = hyperedge) and the hyperedge count.
    """
    B, T = umask.shape
    dev = umask.device
    V = 1 if virtual else 0
    M = n_modals
    P = M + V                                              # node planes per utterance
    ok = umask.bool()

    spk = qmask.argmax(-1)
    S = qmask.size(-1)
    if chain == 'time':
        assert shift_adj is not None, "chain='time' needs shift_adj"
        tb = time_block_ids(shift_adj, umask)              # [B, T]
        chains = [(1, tb)]                                  # a single temporal chain
        Kmax = T + 2
    else:
        blk = block_ids(shift_pred, qmask, prev, valid)    # [B, T, 2]
        chains = [(0, spk * (T + 2) + blk[..., 0]), (1, blk[..., 1])]
        Kmax = max(S * (T + 2), T + 2) + 1

    b_idx = torch.arange(B, device=dev)[:, None].expand(B, T)
    t_idx = torch.arange(T, device=dev)[None, :].expand(B, T)

    planes = P if virtual_in_segment else M                 # does the virtual node get context?
    node_src, edge_code = [], []
    for chain_id, key in chains:
        for m in range(planes):
            node = b_idx * P * T + m * T + t_idx            # [B, T]
            code = ((b_idx * 2 + chain_id) * Kmax + key) * (P + 1) + m
            node_src.append(node[ok])
            edge_code.append(code[ok])

    if virtual:
        # one hyperedge per utterance: all modalities plus the virtual node
        base = (B * 2 * Kmax) * (P + 1)                     # keep the code space disjoint
        for m in range(P):
            node = b_idx * P * T + m * T + t_idx
            code = base + b_idx * T + t_idx
            node_src.append(node[ok])
            edge_code.append(code[ok])

    node_src = torch.cat(node_src)
    edge_code = torch.cat(edge_code)
    _, edge_id = torch.unique(edge_code, return_inverse=True)
    return torch.stack([node_src, edge_id]), int(edge_id.max()) + 1 if edge_id.numel() else 0


# ---------------------------------------------------------------------------
# Hypergraph convolution
# ---------------------------------------------------------------------------
class HyperConv(nn.Module):
    def __init__(self, in_dim, out_dim, attn=True, dropout=0.0, loo=True):
        super().__init__()
        self.attn, self.loo = attn, loo
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = dropout
        if attn:
            self.score = nn.Linear(out_dim, 1)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x, edge_index, n_edges, w_e=None):
        """x [n_nodes, in_dim] -> [n_nodes, out_dim]"""
        n_nodes = x.size(0)
        v, e = edge_index
        h = self.W(F.dropout(x, self.dropout, self.training))

        # ---- gather: unnormalised sums per hyperedge, so leave-one-out stays exact
        if self.attn:
            s = self.score(h)[v].squeeze(-1)                                  # [nnz]
            hi = torch.full((n_edges,), -1e30, device=x.device).index_reduce_(
                0, e, s, 'amax', include_self=True)
            a = (s - hi[e]).exp()                                             # unnormalised weight
        else:
            a = torch.ones(v.numel(), device=x.device, dtype=h.dtype)
        S = torch.zeros(n_edges, h.size(-1), device=x.device).index_add_(0, e, a[:, None] * h[v])
        A = torch.zeros(n_edges, device=x.device).index_add_(0, e, a)

        # ---- message of edge e to node v, node-specific when loo
        if self.loo:
            num, den = S[e] - a[:, None] * h[v], (A[e] - a).clamp(min=1e-12)
            msg = num / den[:, None]
            alone = (A[e] - a) <= 1e-9                    # v is the only member: nothing to say
            msg = torch.where(alone[:, None], torch.zeros_like(msg), msg)
        else:
            msg = S[e] / A[e].clamp(min=1e-12)[:, None]

        # ---- scatter: hyperedges -> nodes, normalised by the node's hyperedge weight sum
        w = torch.ones(n_edges, device=x.device) if w_e is None else w_e
        out = torch.zeros(n_nodes, h.size(-1), device=x.device).index_add_(
            0, v, w[e][:, None] * msg)
        deg = torch.zeros(n_nodes, device=x.device).index_add_(0, v, w[e])
        return out / deg[:, None].clamp(min=1e-12) + self.bias


class EmotionalHyperGraph(nn.Module):
    """features (list of M x [B, T, H]) -> list of (M + 1) x [B, T, H] (last one is the virtual node)"""

    def __init__(self, hidden_dim, n_modals=3, layers=1, dropout=0.1, attn=True, loo=True,
                 virtual=True, virtual_in_segment=True, soft_weight=False, init_mu=0.1):
        super().__init__()
        self.n_modals, self.dropout = n_modals, dropout
        self.virtual, self.virtual_in_segment = virtual, virtual_in_segment
        self.soft_weight = soft_weight
        self.layers = nn.ModuleList([HyperConv(hidden_dim, hidden_dim, attn, dropout, loo)
                                     for _ in range(layers)])
        if soft_weight:
            import math
            theta0 = math.log(math.expm1(init_mu)) if init_mu > 0 else -20.0
            self.theta_mu = nn.Parameter(torch.tensor(theta0))

    @property
    def mu(self):
        return F.softplus(self.theta_mu) if self.soft_weight else None

    def forward(self, features, qmask, umask, shift_pred, prev, valid, cons=None,
                chain='both', shift_adj=None,
                virtual_node=None):
        B, T, H = features[0].shape
        M = self.n_modals
        P = M + (1 if self.virtual else 0)

        planes = list(features)
        if self.virtual:
            if virtual_node is None:
                virtual_node = torch.stack(features, 0).mean(0)
            elif virtual_node.shape != (B, T, H):
                raise ValueError(
                    f'virtual_node must have shape {(B, T, H)}, got {tuple(virtual_node.shape)}')
            planes.append(virtual_node)
        x = torch.cat(planes, dim=1).reshape(B * P * T, H)

        ei, n_edges = build_segment_hyperedges(qmask, umask, shift_pred, prev, valid, M,
                                               self.virtual, self.virtual_in_segment,
                                               chain=chain, shift_adj=shift_adj)
        for layer in self.layers:
            x = x + F.elu(F.dropout(layer(x, ei, n_edges), self.dropout, self.training))
        return list(x.reshape(B, P * T, H).split(T, dim=1))

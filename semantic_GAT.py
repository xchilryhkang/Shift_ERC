"""
Semantic context graph with a structural log-prior (dense implementation, no torch_geometric).
 
Graph (per dialogue of length T, modalities M = e.g. [t, a, v]):
    nodes  : v_i^m, one per (utterance i, modality m)            -> N = M * T nodes
    edges  : j -> i (source j, target i), four relations
        self  : v_i^m  -> v_i^m
        mod   : v_i^m' -> v_i^m,  m' != m                         (same utterance)
        same  : v_j^m  -> v_i^m,  i-d <= j <= i-1, s_j == s_i     (past only)
        cross : v_j^m  -> v_i^m,  i-d <= j <= i-1, s_j != s_i     (past only)
    optional: always add the edge p_u(i) -> i (nearest previous utterance of the same
              speaker) even if it lies outside the window d (useful for MELD).
 
Attention of layer, head h:
    c_ji   = LeakyReLU(a_src^h . W^h h_j + a_dst^h . W^h h_i)          content score
    rho_ji = b^h_{r(j,i)} - lambda^h * phi(Delta_ji)                     structural log-prior
             phi(Delta) = max(Delta - 1, 0),  Delta = i - j  (0 for self / mod edges)
             b^h_self = 0 (reference),  lambda^h = softplus(theta^h) >= 0
    alpha_ji = softmax_j( c_ji + rho_ji )
    h'_i   = ||_h sum_j alpha_ji W^h h_j + bias
 
Equivalently alpha_ji ∝ exp(c_ji) * pi_ji with pi_ji = exp(b_r) * exp(-lambda * phi(Delta)).
With b = 0 and lambda = 0 the layer reduces to a plain GAT on the same edges.
 
Update (as in ESDCM):
    h~ = h + ELU(Dropout(GATLayer(Dropout(h))))           per layer (residual)
    out = sigmoid(W1 h0) * h0 + sigmoid(W2 h~) * h~      gate between input and graph output
 
Tensor conventions follow the baseline repo:
    features : list of M tensors [B, T, H]
    qmask    : [B, T, n_speakers] one-hot (already permuted in train.py)
    umask    : [B, T] 1 = real utterance, 0 = padding
"""


import math
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
REL_NONE, REL_SELF, REL_MOD, REL_SAME, REL_CROSS = 0, 1, 2, 3, 4
REL_NAMES = {REL_SELF: 'self', REL_MOD: 'mod', REL_SAME: 'same', REL_CROSS: 'cross'}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_relation_graph(qmask, umask, n_modals, window=4, link_prev_same=False):
    """
    Returns
        rel : [B, N, N] long, rel[b, i, j] = relation of edge j -> i (REL_NONE if no edge)
        phi : [N, N]    float, phi[i, j] = max(Delta - 1, 0) for context edges, 0 otherwise
    window = -1 keeps every past utterance of the dialogue, so the distance decay
    -lambda * phi becomes the only thing limiting the receptive field.
    Node index n = m * T + t  (modality block, then time).
    Row = target (query), column = source (key).
    """
    B, T = umask.shape
    device = umask.device
    N = n_modals * T

    n = torch.arange(N, device=device)
    mod_idx, t_idx = n // T, n % T                                   # [N]
    same_mod = mod_idx[:, None] == mod_idx[None, :]                  # [N, N]
    same_time = t_idx[:, None] == t_idx[None, :]
    delta = t_idx[:, None] - t_idx[None, :]                          # target - source

    is_self = same_mod & same_time
    is_mod = (~same_mod) & same_time
    in_window = same_mod & (delta >= 1)                              # past only
    if window >= 0:
        in_window = in_window & (delta <= window)

    valid = umask.bool()                                             # [B, T]
    key_valid = valid[:, t_idx][:, None, :]                          # [B, 1, N]

    spk = qmask.argmax(dim=-1)                                       # [B, T]
    spk_n = spk[:, t_idx]                                            # [B, N]
    same_spk = spk_n[:, :, None] == spk_n[:, None, :]                # [B, N, N]

    ctx = in_window.unsqueeze(0).expand(B, N, N)
    if link_prev_same:
        # p_u(i): nearest previous utterance of the same speaker
        tt = torch.arange(T, device=device)
        spk_eq = spk[:, :, None] == spk[:, None, :]                   # [B, T, T]
        cand = spk_eq & (tt[None, None, :] < tt[None, :, None]) & valid[:, None, :]
        pu = torch.where(cand, tt[None, None, :], torch.full_like(cand, -1, dtype=torch.long)).max(-1).values
        pu_n = pu[:, t_idx]                                          # [B, N]  (-1 if none)
        prev_edge = same_mod[None] & (t_idx[None, None, :] == pu_n[:, :, None])
        ctx = ctx | prev_edge
 
    ctx = ctx & key_valid
 
    rel = torch.zeros(B, N, N, dtype=torch.long, device=device)
    rel[(is_mod.unsqueeze(0) & key_valid)] = REL_MOD
    rel[ctx & same_spk] = REL_SAME
    rel[ctx & ~same_spk] = REL_CROSS
    rel[is_self.unsqueeze(0).expand(B, N, N)] = REL_SELF             # always present -> no empty rows
 
    phi = (delta - 1).clamp(min=0).float() * (delta >= 1).float()    # 0 for self / mod
    return rel, phi


class StructuralPriorGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, negative_slope=0.2, attn_dropout=0.0, init_lambda=0.5, learn_prior=True):
        super().__init__()
        assert out_dim % heads == 0, 'out_dim must be divisible by heads'

        self.heads, self.C = heads, out_dim // heads
        self.negative_slope = negative_slope
        self.attn_dropout = attn_dropout

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.C))
        self.a_dst = nn.Parameter(torch.empty(heads, self.C))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # structural prior: b for (mod, same, cross); b_self = 0 is the reference
        self.b_rel = nn.Parameter(torch.zeros(3, heads), requires_grad=learn_prior)
        theta0 = math.log(math.expm1(init_lambda)) if init_lambda > 0 else -20.0   # softplus^-1
        self.theta = nn.Parameter(torch.full((heads,), theta0), requires_grad=learn_prior)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    @property
    def lamda(self):
        return F.softplus(self.theta) # [heads]

    def rel_bias_table(self):
        # rows indexed by relation id: none, self, mod, same, cross
        zero = self.b_rel.new_zeros(1, self.heads)
        return torch.cat([zero, zero, self.b_rel], dim=0)  # [5, heads]

    def forward(self, x, rel, phi, return_attention=False):
        """x: [B, N, in_dim], rel: [B, N, N], phi: [N, N]"""
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.C) # [B, N, H, C]

        s_src = (h * self.a_src).sum(-1) # [B, N, H]
        s_dst = (h * self.a_dst).sum(-1)
        content = F.leaky_relu(s_dst[:, :, None, :] + s_src[:, None, :, :], self.negative_slope)

        prior = self.rel_bias_table()[rel] - self.lamda * phi[None, :, :, None] # [B, N, N, H]
        energy = (content + prior).masked_fill((rel == REL_NONE)[..., None], float("-inf"))

        alpha = torch.softmax(energy, dim=2)
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

        out = torch.einsum('bijh,bjhc->bihc', alpha, h).reshape(B, N, self.heads * self.C) + self.bias
        return (out, alpha) if return_attention else out


class SemanticContextGraph(nn.Module):
    """
    features (list of M x [B, T, H] -> list of M x [B, T, H])
    """
    def __init__(self, hidden_dim, n_modals=3, heads=4, layers=1, window=4, link_prev_same=False, dropout=0.1, attn_dropout=0.0, init_lambda=0.5, learn_prior=True, gate=True):
        super().__init__()
        self.n_modals, self.window, self.link_prev_same = n_modals, window, link_prev_same
        self.dropout = dropout
        self.layers = nn.ModuleList([
            StructuralPriorGATLayer(hidden_dim, hidden_dim, heads=heads, attn_dropout=attn_dropout, init_lambda=init_lambda, learn_prior=learn_prior)
            for _ in range(layers)
        ])

        self.gate = gate
        if gate: 
            self.gate_in = nn.Linear(hidden_dim, hidden_dim)
            self.gate_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features, qmask, umask, return_attention=False, return_layers=False):
        assert len(features) == self.n_modals
        B, T, H = features[0].shape 
        rel, phi = build_relation_graph(qmask, umask, self.n_modals, self.window, self.link_prev_same)

        x0 = torch.cat(features, dim=1) # [B, M*T, H]
        x = F.dropout(x0, p=self.dropout, training=self.training)
        attns, per_layer = [], [x0]          # per_layer[0] is the input to the graph
        for layer in self.layers:
            out, a = layer(x, rel, phi, return_attention=True)
            x = x + F.elu(F.dropout(out, p=self.dropout, training=self.training))
            attns.append(a)
            per_layer.append(x)

        if self.gate: 
            x = torch.sigmoid(self.gate_in(x0)) * x0 + torch.sigmoid(self.gate_out(x)) * x
        per_layer.append(x)                  # after the gate

        outs = list(x.split(T, dim=1)) # M x [B, T, H]
        if return_attention or return_layers:
            info = {"attention": attns, "relation": rel, "phi": phi}
            if return_layers:
                info["layers"] = per_layer   # [input, after L1, ..., after Ln, after gate]
            return outs, info
        return outs


    @torch.no_grad()
    def prior_summary(self):
        """Learned prior per layer / head: b_mod, b_same, b_cross, lambda (for interpretation)."""
        rows = []
        for l, layer in enumerate(self.layers):
            for h in range(layer.heads):
                b = layer.b_rel[:, h].tolist()
                rows.append(dict(layer=l, head=h, b_mod=b[0], b_same=b[1], b_cross=b[2],
                                 lam=layer.lamda[h].item()))
        return rows


# ---------------------------------------------------------------------------
# Sanity checks:  python graph.py
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    torch.manual_seed(0)
 
    # 1) hand example: dialogue A B A B | A, target = text node of u5 (index 4), window 4.
    #    Content scores switched off (a_src = a_dst = 0) -> attention = normalized prior.
    B, T, H, M = 1, 5, 8, 3
    qmask = torch.zeros(B, T, 2)
    for t in range(T):
        qmask[0, t, t % 2] = 1                                       # A, B, A, B, A
    umask = torch.ones(B, T)
    layer = StructuralPriorGATLayer(H, H, heads=1, init_lambda=0.5)
    with torch.no_grad():
        layer.a_src.zero_(); layer.a_dst.zero_()
        layer.b_rel.copy_(torch.tensor([[-0.5], [0.3], [-0.4]]))    # b_mod, b_same, b_cross
    rel, phi = build_relation_graph(qmask, umask, M, window=4)
    _, alpha = layer(torch.randn(B, M * T, H), rel, phi, return_attention=True)
    tgt = 4                                                           # text block, u5
    got = {j: round(alpha[0, tgt, j, 0].item(), 3) for j in range(M * T) if rel[0, tgt, j] != REL_NONE}
    expected_pi = {4: 1.0, 9: math.exp(-0.5), 14: math.exp(-0.5),    # self, mod(a), mod(v)
                   3: math.exp(-0.4), 2: math.exp(0.3 - 0.5),        # u4 cross, u3 same
                   1: math.exp(-0.4 - 1.0), 0: math.exp(0.3 - 1.5)}  # u2 cross, u1 same
    Z = sum(expected_pi.values())
    exp_alpha = {j: round(p / Z, 3) for j, p in expected_pi.items()}
    print('attention of u5 (text):', got)
    assert got == exp_alpha, (got, exp_alpha)
 
    # 2) padding, length-1 dialogues, multi-party, link_prev_same: shapes, no NaN, gradients
    B, T, H, S = 3, 9, 16, 9
    lengths = [9, 4, 1]
    umask = torch.zeros(B, T); qmask = torch.zeros(B, T, S)
    for b, L in enumerate(lengths):
        umask[b, :L] = 1
        for t in range(L):
            qmask[b, t, torch.randint(0, S, (1,))] = 1
    g = SemanticContextGraph(H, n_modals=3, heads=4, layers=2, window=2, link_prev_same=True)
    feats = [torch.randn(B, T, H, requires_grad=True) for _ in range(3)]
    outs, info = g(feats, qmask, umask, return_attention=True)
    assert len(outs) == 3 and all(o.shape == (B, T, H) for o in outs)
    assert all(torch.isfinite(o).all() for o in outs)
    sum(o[umask.bool()].sum() for o in outs).backward()
    assert g.layers[0].b_rel.grad is not None and g.layers[0].theta.grad is not None
 
    # 3) no edge from a padded utterance into a real one, and context edges only point to the past
    rel = info['relation']
    t_idx = torch.arange(3 * T) % T
    for b, L in enumerate(lengths):
        real_q = t_idx < L
        pad_k = t_idx >= L
        assert (rel[b][real_q][:, pad_k] == REL_NONE).all()
    ctx = (rel == REL_SAME) | (rel == REL_CROSS)
    assert (t_idx[None, None, :] < t_idx[None, :, None]).expand_as(ctx)[ctx].all()
    print('all checks passed')
    print(g.prior_summary()[:2])
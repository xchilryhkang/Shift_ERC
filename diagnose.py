"""
Diagnose, on a trained checkpoint:
  (2) how well the contrastive loss shaped the graph-1 space, and
  (3) how good the partition the cosine cut produces actually is.

Nothing is retrained; one forward pass over the test set feeds both parts.

(2) contrastive space, on h1 (graph-1 output, before fusion):
      silhouette      cluster quality by emotion label   (-1..1, higher = cleaner clusters)
      inter/intra     mean cosine distance between different-emotion pairs over same-emotion
                      pairs; > 1 means emotions are pulled apart
      AUC(cos)        ROC-AUC of predicting shift from 1 - cos(h_zeta, h_i) alone
      per polarity    the same inter/intra split by 3-polarity, to see if the space separates
                      polarity better than fine emotion (or the reverse)

(3) cosine-cut partition, swept over cut_tau, vs the oracle (ground-truth) partition:
      boundary F1     do the predicted cut points match the true polarity/emotion changes
      edge purity     fraction of connected pairs that truly share the label
      isolated        utterances left with no context edge
      block mean      average block size
    reported for both the polarity oracle and the full-emotion oracle, per perspective.

Usage
    python diagnose.py --Dataset IEMOCAP --data_dir data --checkpoint ./C/bestModel.pth \
        --hidden_dim 512 --layers 4 --heads 4 --window -1 --no_shift
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, silhouette_score
from torch.utils.data import DataLoader

from contrastive import cosine_shift
from dataloader import IEMOCAPDataset, MELDDataset
from model import BaselineModel
from sheaf import block_ids
from shift import POLARITY, build_shift_pairs, polarity_map

TAUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def gather_pairs(z, prev):
    z_i = z.unsqueeze(2).expand(-1, -1, 2, -1)
    z_s = torch.gather(z_i, 1, prev.unsqueeze(-1).expand(-1, -1, -1, z.size(-1)))
    return z_s, z_i


def inter_intra(z_flat, y_flat):
    """mean cross-label cosine distance / mean same-label cosine distance."""
    zn = torch.nn.functional.normalize(z_flat, dim=-1)
    d = 1 - zn @ zn.t()                                        # cosine distance
    same = y_flat[:, None] == y_flat[None, :]
    eye = torch.eye(len(y_flat), dtype=torch.bool)
    same = same & ~eye
    diff = ~same & ~eye
    return (d[diff].mean() / d[same].mean().clamp(min=1e-6)).item()


def cut_from_cosine(h, qmask, umask, prev, valid, tau):
    return cosine_shift(h, qmask, umask, prev, valid, tau)


def oracle_shift(labels, prev, valid, keymap):
    k = keymap[labels.clamp(min=0)]
    kp = torch.gather(k.unsqueeze(-1).expand(-1, -1, 2), 1, prev)
    return (kp != k.unsqueeze(-1)) & valid


def partition_quality(h, qmask, umask, labels, prev, valid, tau, keymap):
    """Compare the cosine cut at tau against the oracle cut defined by keymap."""
    sp = cut_from_cosine(h, qmask, umask, prev, valid, tau)
    orc = oracle_shift(labels, prev, valid, keymap)
    m = valid
    bf1 = f1_score(orc[m].cpu().numpy(), sp[m].cpu().numpy(), zero_division=0)

    rel, _ = _seg_edges(qmask, umask, sp, prev, valid)
    ctx = (rel == 2) | (rel == 3)
    ok = umask.bool()
    deg = ctx.sum(-1)[ok]
    k = keymap[labels.clamp(min=0)]
    same = k[:, :, None] == k[:, None, :]
    edges = int(ctx.sum())
    pure = int((ctx & same).sum()) / max(edges, 1)
    return dict(bf1=bf1 * 100, iso=100 * (deg == 0).float().mean().item(),
                blk=deg.float().mean().item() + 1, pure=pure * 100)


def _seg_edges(qmask, umask, sp, prev, valid):
    from sheaf import build_segment_edges
    return build_segment_edges(qmask, umask, sp, prev, valid, n_modals=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--Dataset', default='IEMOCAP')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--hidden_dim', type=int, default=512)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--window', type=int, default=-1)
    ap.add_argument('--modals', default='tav')
    ap.add_argument('--no_shift', action='store_true')
    ap.add_argument('--no-cuda', action='store_true')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    if args.Dataset == 'IEMOCAP':
        ds = IEMOCAPDataset(f'{args.data_dir}/iemocap_multimodal_features.pkl', train=False)
        D_a, n_cls = 1582, 6
    else:
        ds = MELDDataset(f'{args.data_dir}/meld_multimodal_features.pkl', train=False)
        D_a, n_cls = 300, 7
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=ds.collate_fn)

    model = BaselineModel(1024, 342, D_a, n_classes=n_cls, hidden_dim=args.hidden_dim, dropout=0.0,
                          modals=args.modals, dataset=args.Dataset, use_graph=True,
                          use_shift=not args.no_shift, heads=args.heads, layers=args.layers,
                          window=args.window).to(dev).eval()
    miss = model.load_state_dict(torch.load(args.checkpoint, map_location=dev), strict=False)
    print(f'checkpoint loaded ({len(miss.missing_keys)} missing, '
          f'{len(miss.unexpected_keys)} unexpected keys)')

    pol = polarity_map(args.Dataset, dev)
    emo = torch.arange(n_cls, device=dev)

    H, Y, cache = [], [], []
    with torch.no_grad():
        for data in loader:
            textf, visuf, acouf, qmask, umask, label = [d.to(dev) for d in data]
            qmask = qmask.permute(1, 0, 2)
            inputs = {'t': textf, 'a': acouf, 'v': visuf}
            xs = [model.proj[m](inputs[m].permute(1, 0, 2)) for m in model.modals]
            h1 = sum(model.graph(xs, qmask, umask)) if model.use_graph else sum(xs)
            prev, valid = build_shift_pairs(qmask, umask)
            ok = umask.bool()
            H.append(h1[ok].cpu()); Y.append(label[ok].cpu())
            cache.append((h1, qmask, umask, label, prev, valid))

    Hf, Yf = torch.cat(H), torch.cat(Y)

    # ---- (2) contrastive space
    print('\n=== (2) contrastive space (on h1) ===')
    zf = torch.nn.functional.normalize(Hf, dim=-1).numpy()
    print(f'  silhouette (emotion)  {silhouette_score(zf, Yf.numpy()):.3f}')
    print(f'  silhouette (polarity) {silhouette_score(zf, pol.cpu()[Yf].numpy()):.3f}')
    print(f'  inter/intra (emotion)  {inter_intra(Hf, Yf):.3f}')
    print(f'  inter/intra (polarity) {inter_intra(Hf, pol.cpu()[Yf]):.3f}')
    au, ay = [], []
    for h1, qmask, umask, label, prev, valid in cache:
        z = torch.nn.functional.normalize(h1, dim=-1)
        zs, zi = gather_pairs(z, prev)
        d = 1 - (zs * zi).sum(-1)
        p = pol[label.clamp(min=0)]
        y = (torch.gather(p.unsqueeze(-1).expand(-1, -1, 2), 1, prev) != p.unsqueeze(-1))
        au += d[valid].cpu().tolist(); ay += y[valid].cpu().tolist()
    print(f'  AUC(cos) for shift     {roc_auc_score(ay, au):.3f}')

    # ---- (3) cosine-cut partition vs oracle
    for name, keymap in (('polarity', pol), ('emotion', emo)):
        print(f'\n=== (3) cosine cut vs {name} oracle ===')
        print(f"  {'tau':>5} {'boundaryF1':>11} {'edgePure%':>10} {'isolated%':>10} {'blockMean':>10}")
        for tau in TAUS:
            agg = {}
            for h1, qmask, umask, label, prev, valid in cache:
                q = partition_quality(h1, qmask, umask, label, prev, valid, tau, keymap)
                for k, v in q.items():
                    agg.setdefault(k, []).append(v)
            a = {k: np.mean(v) for k, v in agg.items()}
            print(f"  {tau:>5} {a['bf1']:>11.1f} {a['pure']:>10.1f} {a['iso']:>10.1f} {a['blk']:>10.2f}")

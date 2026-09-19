"""
Is the feature after graph 1 too smooth for shift detection?

The shift head reads h, the output of a 4-layer graph whose job is to blend context. Shift
detection needs the opposite: contrast between two adjacent utterances. This script measures
that directly on a trained checkpoint, layer by layer, with no retraining.

Four measurements, each computed on the graph input (x, right after the Linear projections),
on every intermediate layer, and on the final output (after the gate):

  contrast     ||h_i - h_zeta|| / ||h_i||, averaged over pairs, split into shift / no-shift.
               The gap between the two is the signal a shift head can use.
  AUC(cos)     ROC-AUC of predicting "shift" from the cosine distance alone. No parameters,
               so it measures how much shift information the representation itself carries.
  intra-sim    mean cosine similarity between all utterance pairs of a dialogue. Rising with
               depth is the classic over-smoothing signature.
  probe        accuracy and F1 of a logistic regression trained on [z_zeta || z_i || z_i-z_zeta]
               to predict the binary shift. The strongest evidence: it says which layer a head
               could do best from, independently of the head currently in use.

Usage
    python smoothness.py --Dataset IEMOCAP --data_dir data --checkpoint IEMOCAP/bestModel.pth \
        --hidden_dim 512 --layers 4 --heads 4 --window -1 --shift_depth 1 --shift_compare cat
"""
import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from dataloader import IEMOCAPDataset, MELDDataset
from model import BaselineModel
from shift import build_shift_pairs, polarity_map


def pair_feats(z, prev):
    """z [B, T, H], prev [B, T, 2] -> source and target, each [B, T, 2, H]."""
    z_i = z.unsqueeze(2).expand(-1, -1, 2, -1)
    z_s = torch.gather(z_i, 1, prev.unsqueeze(-1).expand(-1, -1, -1, z.size(-1)))
    return z_s, z_i


def intra_sim(z, umask):
    """Mean cosine similarity between distinct utterances of the same dialogue."""
    zn = torch.nn.functional.normalize(z, dim=-1)
    sim = zn @ zn.transpose(1, 2)                                   # [B, T, T]
    ok = umask.bool()
    pair = ok[:, :, None] & ok[:, None, :]
    eye = torch.eye(z.size(1), dtype=torch.bool, device=z.device)
    pair = pair & ~eye[None]
    return sim[pair].mean().item(), int(pair.sum())


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
    ap.add_argument('--link_prev_same', action='store_true')
    ap.add_argument('--shift_depth', type=int, default=0)
    ap.add_argument('--shift_emo_dim', type=int, default=None)
    ap.add_argument('--shift_compare', default='full', choices=['cat', 'full'])
    ap.add_argument('--shift_mode', default='pair', choices=['pair', 'polarity', 'both'])
    ap.add_argument('--no_shift', action='store_true', help='checkpoint has no shift head (cosine cut)')
    ap.add_argument('--modals', default='tav')
    ap.add_argument('--probe_C', type=float, default=1.0)
    ap.add_argument('--no-cuda', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    if args.Dataset == 'IEMOCAP':
        ds = IEMOCAPDataset(f'{args.data_dir}/iemocap_multimodal_features.pkl', train=False)
        D_a, n_cls = 1582, 6
    else:
        ds = MELDDataset(f'{args.data_dir}/meld_multimodal_features.pkl', train=False)
        D_a, n_cls = 300, 7
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=ds.collate_fn)

    model = BaselineModel(1024, 342, D_a, n_classes=n_cls, hidden_dim=args.hidden_dim, dropout=0.0,
                          modals=args.modals, dataset=args.Dataset, use_graph=True, use_shift=not args.no_shift,
                          heads=args.heads, layers=args.layers, window=args.window,
                          link_prev_same=args.link_prev_same, shift_depth=args.shift_depth,
                          shift_emo_dim=args.shift_emo_dim, shift_compare=args.shift_compare,
                          shift_mode=args.shift_mode).to(device).eval()
    miss = model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    print(f'checkpoint loaded ({len(miss.missing_keys)} missing, '
          f'{len(miss.unexpected_keys)} unexpected keys)')
    pol = polarity_map(args.Dataset, device)

    n_stage = args.layers + 2                    # input, L1..Ln, after gate
    stages = ['x (input)'] + [f'after L{i+1}' for i in range(args.layers)] + ['after gate']
    acc = [dict(c_sh=[], c_no=[], cos=[], y=[], sim=0.0, nsim=0, feat=[]) for _ in range(n_stage)]

    with torch.no_grad():
        for data in loader:
            textf, visuf, acouf, qmask, umask, label = [d.to(device) for d in data]
            qmask = qmask.permute(1, 0, 2)
            inputs = {'t': textf, 'a': acouf, 'v': visuf}
            xs = [model.proj[m](inputs[m].permute(1, 0, 2)) for m in model.modals]
            _, info = model.graph(xs, qmask, umask, return_layers=True)

            prev, valid = build_shift_pairs(qmask, umask)
            p = pol[label.clamp(min=0)]
            y = (torch.gather(p.unsqueeze(-1).expand(-1, -1, 2), 1, prev) != p.unsqueeze(-1))
            T = umask.size(1)

            for s, feat in enumerate(info['layers']):
                z = sum(feat.split(T, dim=1))                        # fuse modalities, as the model does
                z_s, z_i = pair_feats(z, prev)
                d = (z_i - z_s).norm(dim=-1) / z_i.norm(dim=-1).clamp(min=1e-8)
                cos = torch.nn.functional.cosine_similarity(z_s, z_i, dim=-1)
                m, yy = valid, y[valid]
                acc[s]['c_sh'] += d[m][yy].cpu().tolist()
                acc[s]['c_no'] += d[m][~yy].cpu().tolist()
                acc[s]['cos'] += (1 - cos)[m].cpu().tolist()
                acc[s]['y'] += yy.cpu().tolist()
                sm, n = intra_sim(z, umask)
                acc[s]['sim'] += sm * n
                acc[s]['nsim'] += n
                acc[s]['feat'].append(torch.cat([z_s, z_i, z_i - z_s], -1)[m].cpu().numpy())

    print(f"\n{'stage':>12} {'contrast(shift)':>16} {'contrast(no)':>13} {'gap':>7} "
          f"{'AUC(cos)':>9} {'intra-sim':>10} {'probe acc':>10} {'probe F1':>9}")
    print('-' * 92)
    for s in range(n_stage):
        a = acc[s]
        y = np.array(a['y'])
        X = np.concatenate(a['feat'])
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        n_tr = int(0.7 * len(y))
        clf = LogisticRegression(max_iter=2000, C=args.probe_C).fit(X[:n_tr], y[:n_tr])
        pred = clf.predict(X[n_tr:])
        c_sh, c_no = np.mean(a['c_sh']), np.mean(a['c_no'])
        print(f"{stages[s]:>12} {c_sh:16.3f} {c_no:13.3f} {c_sh - c_no:7.3f} "
              f"{roc_auc_score(y, a['cos']):9.3f} {a['sim'] / a['nsim']:10.3f} "
              f"{(pred == y[n_tr:]).mean() * 100:10.2f} "
              f"{f1_score(y[n_tr:], pred, zero_division=0) * 100:9.2f}")
    print(f"\nshift rate: {100 * np.mean(acc[0]['y']):.1f}%   probe trained on the first 70% of "
          f"the test pairs, scored on the rest")

"""
Sweep the shift threshold tau on a trained checkpoint, without retraining.

tau only changes how the 9-class distribution is turned into a binary "is there a shift"
decision (see sheaf.predict_shift), so every value can be evaluated from the same forward
pass. For each tau the script reports the binary quality of the shift decision and what the
partition looks like, so you can pick tau before rerunning training with it.

Usage
    python sweep_tau.py --Dataset IEMOCAP --data_dir data --checkpoint IEMOCAP/bestModel.pth \
        --hidden_dim 512 --layers 4 --window -1 --heads 4 --shift_depth 1 --shift_compare cat
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from dataloader import IEMOCAPDataset, MELDDataset
from model import BaselineModel
from sheaf import build_segment_edges, consistency
from shift import build_shift_pairs, polarity_map

TAUS = [None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def partition_stats(qmask, umask, shift_pred, prev, valid, labels, pol):
    """Context edges per utterance, isolated ratio, and edge purity, on utterance nodes only."""
    rel, _ = build_segment_edges(qmask, umask, shift_pred, prev, valid, n_modals=1)
    ctx = (rel == 2) | (rel == 3)                       # REL_SAME | REL_CROSS
    ok = umask.bool()
    deg = ctx.sum(-1)[ok]
    p = pol[labels.clamp(min=0)]
    same_pol = p[:, :, None] == p[:, None, :]
    return dict(deg=deg.float().sum().item(), n=int(ok.sum()), iso=int((deg == 0).sum()),
                edges=int(ctx.sum()), pure=int((ctx & same_pol).sum()))


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
    ap.add_argument('--shift_mode', default='pair', choices=['pair','polarity','both'])
    ap.add_argument('--modals', default='tav')
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
                          modals=args.modals, dataset=args.Dataset, use_graph=True, use_shift=True,
                          heads=args.heads, layers=args.layers, window=args.window,
                          link_prev_same=args.link_prev_same, shift_depth=args.shift_depth,
                          shift_emo_dim=args.shift_emo_dim,
                          shift_compare=args.shift_compare,
                          shift_mode=args.shift_mode).to(device).eval()
    miss = model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    print(f'checkpoint loaded ({len(miss.missing_keys)} missing, '
          f'{len(miss.unexpected_keys)} unexpected keys)')
    pol = polarity_map(args.Dataset, device)

    # one forward pass; tau only affects post-processing
    cache = []
    with torch.no_grad():
        for data in loader:
            textf, visuf, acouf, qmask, umask, label = [d.to(device) for d in data]
            qmask = qmask.permute(1, 0, 2)
            _, _, _, logits, _ = model(textf, visuf, acouf, umask, qmask, None)
            prev, valid = build_shift_pairs(qmask, umask)
            p = pol[label.clamp(min=0)]
            y_bin = (torch.gather(p.unsqueeze(-1).expand(-1, -1, 2), 1, prev) != p.unsqueeze(-1))
            cache.append((qmask, umask, label, logits, prev, valid, y_bin))

    print(f"\n{'tau':>6} {'acc':>7} {'F1':>7} {'prec':>7} {'rec':>7} "
          f"{'pred%':>7} {'edges/utt':>10} {'iso%':>6} {'purity%':>8}")
    print('-' * 72)
    for tau in TAUS:
        bp, bl, acc = [], [], dict(deg=0, n=0, iso=0, edges=0, pure=0)
        for qmask, umask, label, logits, prev, valid, y_bin in cache:
            sp = (logits.argmax(-1) // 3 != logits.argmax(-1) % 3) if tau is None \
                else consistency(logits) < tau
            bp.append(sp[valid].cpu().numpy())
            bl.append(y_bin[valid].cpu().numpy())
            for k, v in partition_stats(qmask, umask, sp, prev, valid, label, pol).items():
                acc[k] += v
        bp, bl = np.concatenate(bp), np.concatenate(bl)
        e = max(acc['edges'], 1)
        print(f"{str(tau):>6} {100*accuracy_score(bl,bp):7.2f} {100*f1_score(bl,bp,zero_division=0):7.2f} "
              f"{100*precision_score(bl,bp,zero_division=0):7.2f} {100*recall_score(bl,bp,zero_division=0):7.2f} "
              f"{100*bp.mean():7.1f} {acc['deg']/acc['n']:10.2f} "
              f"{100*acc['iso']/acc['n']:6.1f} {100*acc['pure']/e:8.1f}")
    print(f"\ntrue shift rate: {100*bl.mean():.1f}%   (a model that never flags a shift "
          f"still scores {100*(1-bl.mean()):.1f}% accuracy)")

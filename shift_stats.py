"""
Count how often each shift pattern occurs, straight from the ground-truth labels.

No model involved: this is a property of the data, so it tells you what the shift head is
actually being asked to learn, and whether the class imbalance alone explains the weak
minority-pattern F1.

Reports, for train and test separately:
  * the emotion distribution and the polarity distribution
  * the 3x3 polarity-shift matrix, per perspective (p_u = same speaker, q_u = other speaker)
  * the binary shift rate, which is the number an "always no-shift" model would score
  * the M x M emotion-shift matrix (the top entries only)

Usage
    python shift_stats.py --Dataset IEMOCAP --data_dir data
"""
import argparse
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import IEMOCAPDataset, MELDDataset
from shift import POLARITY, build_shift_pairs, polarity_map

NAMES = {'IEMOCAP': ['hap', 'sad', 'neu', 'ang', 'exc', 'fru'],
         'MELD': ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']}
POL_NAMES = ['neg', 'neu', 'pos']


def collect(loader, pol):
    """Returns per-perspective lists of (source emotion, target emotion)."""
    pairs = [[], []]
    emo = Counter()
    for textf, visuf, acouf, qmask, umask, label in loader:
        qmask = qmask.permute(1, 0, 2)
        prev, valid = build_shift_pairs(qmask, umask)
        ok = umask.bool()
        for e in label[ok].tolist():
            emo[e] += 1
        for k in range(2):
            m = valid[..., k]
            src = torch.gather(label, 1, prev[..., k])[m]
            tgt = label[m]
            pairs[k] += list(zip(src.tolist(), tgt.tolist()))
    return pairs, emo


def show_matrix(counts, n, names, title):
    total = counts.sum()
    print(f'\n{title}  (n = {total})')
    print('          ' + ''.join(f'{c:>9}' for c in names) + '      row%')
    for i in range(n):
        row = counts[i].sum()
        print(f'{names[i]:>8}  ' + ''.join(f'{counts[i, j]:>6} {100*counts[i,j]/max(total,1):4.1f}'
                                          for j in range(n)) + f'   {100*row/max(total,1):5.1f}')
    diag = np.trace(counts)
    print(f'{"no shift":>8}: {diag:>6} ({100*diag/max(total,1):.1f}%)    '
          f'shift: {total-diag} ({100*(total-diag)/max(total,1):.1f}%)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--Dataset', default='IEMOCAP')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--top', type=int, default=10, help='how many emotion-shift patterns to list')
    args = ap.parse_args()

    cls = IEMOCAPDataset if args.Dataset == 'IEMOCAP' else MELDDataset
    fn = 'iemocap_multimodal_features.pkl' if args.Dataset == 'IEMOCAP' \
        else 'meld_multimodal_features.pkl'
    names = NAMES[args.Dataset]
    M, n_pol = len(names), 3
    pol = polarity_map(args.Dataset)

    for split, is_train in (('TRAIN', True), ('TEST', False)):
        ds = cls(f'{args.data_dir}/{fn}', train=is_train)
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=ds.collate_fn)
        pairs, emo = collect(loader, pol)

        print(f'\n{"=" * 74}\n{split}   dialogues: {len(ds)}   utterances: {sum(emo.values())}')
        tot = sum(emo.values())
        print('emotions : ' + '  '.join(f'{names[e]} {100*emo[e]/tot:.1f}%' for e in range(M)))
        pc = Counter(POLARITY[args.Dataset][e] for e in emo.elements())
        print('polarity : ' + '  '.join(f'{POL_NAMES[p]} {100*pc[p]/tot:.1f}%' for p in range(n_pol)))

        all_pairs = []
        for k, tag in enumerate(['p_u(i)  same speaker', 'q_u(i)  other speaker']):
            mat = np.zeros((n_pol, n_pol), dtype=int)
            for s, t in pairs[k]:
                mat[POLARITY[args.Dataset][s], POLARITY[args.Dataset][t]] += 1
            show_matrix(mat, n_pol, POL_NAMES, f'polarity shift, {tag}')
            all_pairs += pairs[k]

        mat = np.zeros((n_pol, n_pol), dtype=int)
        for s, t in all_pairs:
            mat[POLARITY[args.Dataset][s], POLARITY[args.Dataset][t]] += 1
        show_matrix(mat, n_pol, POL_NAMES, 'polarity shift, both perspectives pooled')

        cnt = Counter((s, t) for s, t in all_pairs)
        tot_p = sum(cnt.values())
        print(f'\ntop {args.top} emotion-shift patterns out of {M * M}:')
        for (s, t), c in cnt.most_common(args.top):
            mark = '  (no shift)' if POLARITY[args.Dataset][s] == POLARITY[args.Dataset][t] else ''
            print(f'  {names[s]:>4} -> {names[t]:<4} {c:>6}  {100*c/tot_p:5.2f}%{mark}')
        rare = [c for _, c in cnt.items() if c / tot_p < 0.005]
        print(f'  patterns below 0.5%: {len(rare)} of {len(cnt)} seen '
              f'({100*sum(rare)/tot_p:.1f}% of all pairs)')
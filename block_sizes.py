"""
How small do the blocks get if we cut by emotion label instead of by polarity?

Using GROUND-TRUTH labels (the oracle partition), this reports, for the speaker chain and the
time chain separately, and for both cut rules (polarity: neg/neu/pos, and full emotion label):
  * the fraction of ADJACENT same-speaker / same-turn pairs that keep the same label
    (that is exactly the "no shift" rate -- higher means longer blocks)
  * the block-size distribution: mean, median, and the share of singleton blocks
  * how many context edges an utterance would actually get under each rule

A singleton block means the utterance is isolated (only self-loop) unless bidirectional edges
or a coarser level rescue it, so the singleton share is the number to watch.

Usage
    python block_sizes.py --Dataset IEMOCAP --data_dir data
    python block_sizes.py --Dataset MELD --data_dir data
"""
import argparse
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import IEMOCAPDataset, MELDDataset
from shift import POLARITY, build_shift_pairs


def blocks_along(chain_keys, labels, valid_len):
    """
    chain_keys: for each utterance, an id that is constant within a speaker's turns
                (speaker chain) or just the running index (time chain).
    Cut whenever the label changes between consecutive elements of the same chain.
    Returns list of block sizes.
    """
    sizes = []
    for keys, labs in zip(chain_keys, labels):
        # group indices by chain id, keep temporal order
        groups = {}
        for t, k in enumerate(keys):
            groups.setdefault(k, []).append(t)
        for idx in groups.values():
            cur = 1
            for a, b in zip(idx, idx[1:]):
                if labs[a] == labs[b]:
                    cur += 1
                else:
                    sizes.append(cur); cur = 1
            sizes.append(cur)
    return sizes


def adjacent_same_rate(chain_keys, labels):
    """Fraction of consecutive same-chain pairs that share a label."""
    same = tot = 0
    for keys, labs in zip(chain_keys, labels):
        groups = {}
        for t, k in enumerate(keys):
            groups.setdefault(k, []).append(t)
        for idx in groups.values():
            for a, b in zip(idx, idx[1:]):
                tot += 1
                same += int(labs[a] == labs[b])
    return same, tot


def summarize(sizes, tag):
    s = np.array(sizes)
    print(f"  {tag:<22} blocks {len(s):>5} | mean {s.mean():4.2f} | median {np.median(s):.0f} "
          f"| singletons {100*(s==1).mean():4.1f}% | >=3 {100*(s>=3).mean():4.1f}%")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--Dataset', default='IEMOCAP')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()

    cls = IEMOCAPDataset if args.Dataset == 'IEMOCAP' else MELDDataset
    fn = ('iemocap_multimodal_features.pkl' if args.Dataset == 'IEMOCAP'
          else 'meld_multimodal_features.pkl')
    pol = np.array(POLARITY[args.Dataset])

    for split, is_train in (('TRAIN', True), ('TEST', False)):
        ds = cls(f'{args.data_dir}/{fn}', train=is_train)
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=ds.collate_fn)

        spk_keys, time_keys, emo_labels, pol_labels = [], [], [], []
        for _, _, _, qmask, umask, label in loader:
            qmask = qmask.permute(1, 0, 2)
            spk = qmask.argmax(-1)
            for b in range(umask.size(0)):
                L = int(umask[b].sum())
                labs = label[b, :L].tolist()
                sp = spk[b, :L].tolist()
                # speaker chain id = speaker; time chain id = one group (running index)
                spk_keys.append(sp)
                time_keys.append([0] * L)
                emo_labels.append(labs)
                pol_labels.append([int(pol[e]) for e in labs])

        print(f"\n{'='*74}\n{args.Dataset} {split}: {len(emo_labels)} dialogues, "
              f"{sum(len(x) for x in emo_labels)} utterances")

        for lab_name, labs in (('polarity (3)', pol_labels), ('emotion (full)', emo_labels)):
            print(f"\n[{lab_name}]")
            for chain_name, keys in (('speaker chain', spk_keys), ('time chain', time_keys)):
                same, tot = adjacent_same_rate(keys, labs)
                print(f"  {chain_name:<14} adjacent same-label rate (=no-shift): "
                      f"{100*same/max(tot,1):.1f}%  ({same}/{tot})")
                summarize(blocks_along(keys, labs, None), chain_name + ' blocks')
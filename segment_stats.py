"""
Diagnostics for the segment partition.

Answers three questions before spending time on graph 2:
  1. How long are the blocks?  (short -> the partition filters a lot; long -> it barely filters)
  2. How many context edges does an utterance actually get, and how many get none?
  3. How pure are the blocks?  (fraction of connected pairs that really share a polarity)

Everything is reported twice: with the SHIFT HEAD's predictions and with the ORACLE
(ground-truth) shifts. The gap between the two is how much a better shift head could buy;
the oracle row is the ceiling of the whole partition idea.

Usage
    python segment_stats.py --Dataset IEMOCAP --data_dir data                  # untrained head
    python segment_stats.py --Dataset IEMOCAP --data_dir data \
        --checkpoint IEMOCAP/bestModel.pth --hidden_dim 512 --layers 4 --window 6
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import IEMOCAPDataset, MELDDataset
from model import BaselineModel
from sheaf import block_ids, build_segment_edges, predict_shift
from shift import build_shift_pairs, polarity_map


def oracle_shift(labels, prev, valid, pol):
    """Ground-truth shift on each pair: polarity(prev) != polarity(i)."""
    p = pol[labels.clamp(min=0)]                                     # [B, T]
    p_prev = torch.gather(p.unsqueeze(-1).expand(-1, -1, 2), 1, prev)
    return (p_prev != p.unsqueeze(-1)) & valid


def stats(qmask, umask, labels, shift_pred, prev, valid, pol):
    """Returns a dict of partition statistics for one batch."""
    B, T = umask.shape
    ok = umask.bool()
    blk = block_ids(shift_pred, qmask, prev, valid)
    rel, both = build_segment_edges(qmask, umask, shift_pred, prev, valid, n_modals=1,
                                    bidir=BIDIR)
    ctx = (rel == 2) | (rel == 3)                                    # REL_SAME | REL_CROSS

    deg = ctx.sum(-1)[ok]                                            # context edges per utterance
    t = torch.arange(T, device=umask.device)
    spk = qmask.argmax(-1)

    # block sizes: count utterances sharing (speaker, spk_blk) and (time_blk)
    same_spk_blk = (blk[..., 0:1] == blk[..., 0].unsqueeze(1)) & (spk[:, :, None] == spk[:, None, :])
    same_time_blk = blk[..., 1:2] == blk[..., 1].unsqueeze(1)
    pair_ok = ok[:, :, None] & ok[:, None, :]
    len_spk = (same_spk_blk & pair_ok).sum(-1)[ok].float()
    len_time = (same_time_blk & pair_ok).sum(-1)[ok].float()

    # purity: among connected pairs, how many really share a polarity / an emotion
    p = pol[labels.clamp(min=0)]
    same_pol = (p[:, :, None] == p[:, None, :])
    same_emo = (labels[:, :, None] == labels[:, None, :])
    n_edge = ctx.sum().item()
    return dict(
        n_utt=int(ok.sum()),
        n_edge=n_edge,
        deg=deg.float().tolist(),
        isolated=int((deg == 0).sum()),
        len_spk=len_spk.tolist(),
        len_time=len_time.tolist(),
        pure_pol=int((ctx & same_pol).sum()) if n_edge else 0,
        pure_emo=int((ctx & same_emo).sum()) if n_edge else 0,
        shift_rate=float(shift_pred[valid].float().mean()) if valid.any() else 0.0,
    )


def merge(acc, s):
    for k, v in s.items():
        if isinstance(v, list):
            acc.setdefault(k, []).extend(v)
        else:
            acc[k] = acc.get(k, 0) + v
    return acc


def report(acc, n_batch, tag):
    deg = np.array(acc['deg'])
    ls, lt = np.array(acc['len_spk']), np.array(acc['len_time'])
    e = max(acc['n_edge'], 1)
    print(f"\n--- {tag} ---")
    print(f"  utterances                 {acc['n_utt']}")
    print(f"  predicted shift rate       {acc['shift_rate'] / n_batch:.3f}")
    print(f"  block size  speaker chain  mean {ls.mean():.2f}  median {np.median(ls):.0f}  max {ls.max():.0f}")
    print(f"  block size  time chain     mean {lt.mean():.2f}  median {np.median(lt):.0f}  max {lt.max():.0f}")
    print(f"  context edges per utt      mean {deg.mean():.2f}  median {np.median(deg):.0f}  max {deg.max():.0f}")
    print(f"  utterances with no edge    {acc['isolated']} ({100 * acc['isolated'] / acc['n_utt']:.1f}%)")
    print(f"  edge purity (polarity)     {100 * acc['pure_pol'] / e:.1f}%")
    print(f"  edge purity (emotion)      {100 * acc['pure_emo'] / e:.1f}%")


BIDIR = False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--Dataset', default='IEMOCAP')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--checkpoint', default=None, help='bestModel.pth; omit to use an untrained head')
    ap.add_argument('--hidden_dim', type=int, default=512)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--window', type=int, default=4)
    ap.add_argument('--modals', default='tav')
    ap.add_argument('--link_prev_same', action='store_true')
    ap.add_argument('--shift_depth', type=int, default=0)
    ap.add_argument('--shift_emo_dim', type=int, default=None)
    ap.add_argument('--shift_compare', default='full', choices=['cat', 'full'])
    ap.add_argument('--shift_mode', default='pair', choices=['pair', 'polarity', 'both'])
    ap.add_argument('--bidir', action='store_true', help='count edges as --graph2_bidir would')
    ap.add_argument('--no-cuda', action='store_true')
    args = ap.parse_args()
    BIDIR = args.bidir

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
                          shift_emo_dim=args.shift_emo_dim, shift_compare=args.shift_compare,
                          shift_mode=args.shift_mode).to(device).eval()
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location=device)
        missing = model.load_state_dict(sd, strict=False)
        print('checkpoint loaded;', len(missing.missing_keys), 'missing,',
              len(missing.unexpected_keys), 'unexpected keys')
    else:
        print('no checkpoint: the shift head is untrained (random predictions)')

    pol = polarity_map(args.Dataset, device)
    acc_p, acc_o, nb = {}, {}, 0
    with torch.no_grad():
        for data in loader:
            textf, visuf, acouf, qmask, umask, label = [d.to(device) for d in data]
            qmask = qmask.permute(1, 0, 2)
            _, _, _, shift_logits, _ = model(textf, visuf, acouf, umask, qmask, None)
            prev, valid = build_shift_pairs(qmask, umask)
            sp_pred = predict_shift(shift_logits)
            sp_orac = oracle_shift(label, prev, valid, pol)
            acc_p = merge(acc_p, stats(qmask, umask, label, sp_pred, prev, valid, pol))
            acc_o = merge(acc_o, stats(qmask, umask, label, sp_orac, prev, valid, pol))
            nb += 1
            agree = (sp_pred == sp_orac)[valid].float().mean().item()
    report(acc_p, nb, 'PREDICTED shifts')
    report(acc_o, nb, 'ORACLE shifts (ceiling)')
    print(f"\nlast-batch agreement predicted vs oracle: {100 * agree:.1f}%")

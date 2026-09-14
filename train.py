import argparse
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataloader import IEMOCAPDataset, MELDDataset
from model import BaselineModel, MaskedNLLLoss
from shift import build_shift_pairs, shift_labels
from vision import confuPLT


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_loaders(dataset_cls, path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = dataset_cls(path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                              collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler,
                              collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    testset = dataset_cls(path, train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False,
                        w_shift=0.0, warmup=False):
    losses, preds, labels, masks = [], [], [], []
    s_preds, s_labels = [], []

    assert not train or optimizer is not None
    device = next(model.parameters()).device
    model.train() if train else model.eval()

    with torch.set_grad_enabled(train):
        for data in dataloader:
            if train:
                optimizer.zero_grad()

            textf, visuf, acouf, qmask, umask, label = [d.to(device) for d in data]
            qmask = qmask.permute(1, 0, 2)                       # [B, T, n_speakers]
            lengths = umask.sum(dim=1).long()                    # [B]

            log_prob, prob, _, shift_logits = model(textf, visuf, acouf, umask, qmask, lengths,
                                                    warmup=warmup, labels=label)

            lp_ = log_prob.view(-1, log_prob.size(2))
            labels_ = label.view(-1)
            loss = loss_function(lp_, labels_, umask)

            if shift_logits is not None:
                prev, valid = build_shift_pairs(qmask, umask)
                y_shift = shift_labels(label, prev, valid, model.pol)
                loss = loss + w_shift * model.shift.loss(shift_logits, y_shift)
                keep = y_shift.view(-1) != -100
                s_preds.append(shift_logits.reshape(-1, 9).argmax(-1)[keep].cpu().numpy())
                s_labels.append(y_shift.view(-1)[keep].cpu().numpy())

            pred_ = torch.argmax(prob.view(-1, prob.size(2)), 1)
            preds.append(pred_.cpu().numpy())
            labels.append(labels_.cpu().numpy())
            masks.append(umask.view(-1).cpu().numpy())
            losses.append(loss.item() * masks[-1].sum())

            if train:
                loss.backward()
                optimizer.step()

    if not preds:
        return dict(loss=float('nan'), acc=float('nan'), fscore=float('nan'),
                    label=[], pred=[], mask=[], shift_f1=None)

    preds, labels, masks = np.concatenate(preds), np.concatenate(labels), np.concatenate(masks)
    out = dict(loss=round(float(np.sum(losses) / np.sum(masks)), 4),
               acc=round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2),
               fscore=round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2),
               label=labels, pred=preds, mask=masks, shift_f1=None)
    if s_preds:
        sp, sl = np.concatenate(s_preds), np.concatenate(s_labels)
        out['shift_f1'] = round(f1_score(sl, sp, average='macro', zero_division=0) * 100, 2)
        out['shift_acc'] = round(accuracy_score(sl, sp) * 100, 2)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--l2', type=float, default=0.00005, help='weight decay (AdamW)')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=512, help='shared hidden size')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--modals', default='tav', help="subset of 'tav' to use, e.g. t, ta, tav")
    parser.add_argument('--Dataset', default='IEMOCAP', help='IEMOCAP or MELD')
    parser.add_argument('--data_dir', default='data', help='folder containing the *_multimodal_features.pkl files')
    parser.add_argument('--save_model_path', default='./IEMOCAP', type=str, help='output folder')
    parser.add_argument('--seed', default=2094, type=int, help='seed')
    # semantic context graph
    parser.add_argument('--use_graph', action='store_true', help='enable the semantic context graph')
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--window', type=int, default=4, help='context window d (past only)')
    parser.add_argument('--link_prev_same', action='store_true', help='always keep the p_u(i) edge')
    parser.add_argument('--graph_dropout', type=float, default=0.1)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--init_lambda', type=float, default=0.5, help='initial distance decay')
    parser.add_argument('--freeze_prior', action='store_true', help='keep b / lambda fixed (ablation)')
    parser.add_argument('--no_gate', action='store_true')
    parser.add_argument('--prior_lr', type=float, default=0.01, help='lr for the prior scalars')
    # emotion shift
    parser.add_argument('--use_shift', action='store_true', help='enable the 9-way shift head')
    parser.add_argument('--w_shift', type=float, default=0.3, help='weight of the shift loss')
    # emotional context graph (segment partition + sheaf); needs --use_shift
    parser.add_argument('--graph2', default='none', choices=['none', 'gat', 'sheaf'])
    parser.add_argument('--sheaf_d', type=int, default=4, help='stalk dimension (must divide hidden_dim)')
    parser.add_argument('--sheaf_layers', type=int, default=2)
    parser.add_argument('--sheaf_map', default='diag', choices=['diag', 'general'])
    parser.add_argument('--sheaf_step', type=float, default=1.0, help='Euler step size tau')
    parser.add_argument('--graph2_heads', type=int, default=4, help='only for --graph2 gat')
    parser.add_argument('--graph2_layers', type=int, default=1, help='only for --graph2 gat')
    parser.add_argument('--graph2_dropout', type=float, default=0.1)
    parser.add_argument('--oracle_shift', action='store_true',
                        help='build the partition from ground-truth shifts (ceiling experiment)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='epochs with graph2 disabled, while the shift head is still random')
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print('Running on', device)
    os.makedirs(args.save_model_path, exist_ok=True)

    feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024
    n_classes = 6 if args.Dataset == 'IEMOCAP' else 7

    model = BaselineModel(D_text, D_visual, D_audio, n_classes=n_classes, hidden_dim=args.hidden_dim,
                          dropout=args.dropout, modals=args.modals, dataset=args.Dataset,
                          use_graph=args.use_graph, use_shift=args.use_shift,
                          heads=args.heads, layers=args.layers, window=args.window,
                          link_prev_same=args.link_prev_same, graph_dropout=args.graph_dropout,
                          attn_dropout=args.attn_dropout, init_lambda=args.init_lambda,
                          learn_prior=not args.freeze_prior, gate=not args.no_gate,
                          graph2=args.graph2, sheaf_d=args.sheaf_d, sheaf_layers=args.sheaf_layers,
                          sheaf_map=args.sheaf_map, sheaf_step=args.sheaf_step,
                          graph2_heads=args.graph2_heads, graph2_layers=args.graph2_layers,
                          graph2_dropout=args.graph2_dropout, oracle_shift=args.oracle_shift).to(device)
    print(model)
    print('training parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.AdamW(model.param_groups(args.lr, args.prior_lr, args.l2))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()   # same as the original code: no class weights on MELD
        train_loader, valid_loader, test_loader = get_loaders(
            MELDDataset, os.path.join(args.data_dir, 'meld_multimodal_features.pkl'),
            batch_size=args.batch_size, valid=0.0)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1 / 0.086747, 1 / 0.144406, 1 / 0.227883,
                                          1 / 0.160585, 1 / 0.127711, 1 / 0.252668])
        loss_function = MaskedNLLLoss(loss_weights.to(device))
        train_loader, valid_loader, test_loader = get_loaders(
            IEMOCAPDataset, os.path.join(args.data_dir, 'iemocap_multimodal_features.pkl'),
            batch_size=args.batch_size, valid=0.0, pin_memory=True)
    else:
        raise ValueError('Dataset must be IEMOCAP or MELD')

    best, all_test_fscore = None, []

    for e in range(args.epochs):
        start_time = time.time()
        warm = args.graph2 != 'none' and e < args.warmup_epochs
        tr = train_or_eval_model(model, loss_function, train_loader, optimizer, True, args.w_shift, warm)
        te = train_or_eval_model(model, loss_function, test_loader, train=False,
                                 w_shift=args.w_shift, warmup=warm)
        scheduler.step()

        all_test_fscore.append(te['fscore'])
        if best is None or best['fscore'] < te['fscore']:
            best = te
            torch.save(model.state_dict(), os.path.join(args.save_model_path, 'bestModel.pth'))

        msg = ('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, '
               'test_loss: {}, test_acc: {}, test_fscore: {}').format(
            e + 1, tr['loss'], tr['acc'], tr['fscore'], te['loss'], te['acc'], te['fscore'])
        if te['shift_f1'] is not None:
            msg += ', shift_acc: {}, shift_mF1: {}'.format(te['shift_acc'], te['shift_f1'])
        if args.graph2 != 'none':
            msg += ', alpha: {:.3f}'.format(torch.sigmoid(model.alpha).item()) + (' [warmup]' if warm else '')
        print(msg + ', time: {} sec'.format(round(time.time() - start_time, 2)))

        if args.use_graph and not args.freeze_prior and (e + 1) % 10 == 0:
            r = model.graph.prior_summary()[0]
            print('  prior[layer 0, head 0]: b_mod={b_mod:.3f} b_same={b_same:.3f} '
                  'b_cross={b_cross:.3f} lambda={lam:.3f}'.format(**r))

    print('Model Performance:')
    print('Best_Test-FScore-epoch_index: {}'.format(all_test_fscore.index(max(all_test_fscore)) + 1))
    print('Best_Test_F-Score: {}'.format(max(all_test_fscore)))
    print(classification_report(best['label'], best['pred'], sample_weight=best['mask'],
                                digits=4, zero_division=0))
    if args.use_graph:
        import pandas as pd
        print(pd.DataFrame(model.graph.prior_summary()).round(3).to_string(index=False))
    confuPLT(confusion_matrix(best['label'], best['pred'], sample_weight=best['mask']).astype(int),
             args.Dataset, save_path=os.path.join(args.save_model_path, 'heatmap.png'))

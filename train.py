import os
import numpy as np
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, BaselineModel
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import random
from vision import confuPLT
from torch.optim.lr_scheduler import CosineAnnealingLR


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


def train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []

    assert not train or optimizer is not None
    device = next(model.parameters()).device
    model.train() if train else model.eval()

    with torch.set_grad_enabled(train):
        for data in dataloader:
            if train:
                optimizer.zero_grad()

            textf, visuf, acouf, qmask, umask, label = [d.to(device) for d in data]
            qmask = qmask.permute(1, 0, 2)                       # [B, T, n_speakers] (unused by baseline)
            lengths = umask.sum(dim=1).long()                    # [B]

            log_prob, prob, _ = model(textf, visuf, acouf, umask, qmask, lengths)

            lp_ = log_prob.view(-1, log_prob.size(2))
            labels_ = label.view(-1)
            loss = loss_function(lp_, labels_, umask)

            pred_ = torch.argmax(prob.view(-1, prob.size(2)), 1)
            preds.append(pred_.cpu().numpy())
            labels.append(labels_.cpu().numpy())
            masks.append(umask.view(-1).cpu().numpy())
            losses.append(loss.item() * masks[-1].sum())

            if train:
                loss.backward()
                optimizer.step()

    if not preds:
        return float('nan'), float('nan'), [], [], [], float('nan')

    preds, labels, masks = np.concatenate(preds), np.concatenate(labels), np.concatenate(masks)
    avg_loss = round(float(np.sum(losses) / np.sum(masks)), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--l2', type=float, default=0.00005, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=512, metavar='hidden_dim', help='shared hidden size')
    parser.add_argument('--epochs', type=int, default=40, metavar='E', help='number of epochs')
    parser.add_argument('--modals', default='tav', help="subset of 'tav' to use, e.g. t, ta, tav")
    parser.add_argument('--Dataset', default='IEMOCAP', help='IEMOCAP or MELD')
    parser.add_argument('--data_dir', default='data', help='folder containing the *_multimodal_features.pkl files')
    parser.add_argument('--save_model_path', default='./IEMOCAP', type=str, help='output folder')
    parser.add_argument('--seed', default=2094, type=int, help='seed')
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

    model = BaselineModel(D_text, D_visual, D_audio, n_classes=n_classes,
                          hidden_dim=args.hidden_dim, dropout=args.dropout, modals=args.modals).to(device)
    print(model)
    print('training parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

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

    best_fscore, best_label, best_pred, best_mask = None, None, None, None
    all_test_fscore = []

    for e in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(
            model, loss_function, train_loader, optimizer, train=True)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(
            model, loss_function, test_loader, train=False)
        scheduler.step()

        all_test_fscore.append(test_fscore)
        if best_fscore is None or best_fscore < test_fscore:
            best_fscore, best_label, best_pred, best_mask = test_fscore, test_label, test_pred, test_mask
            torch.save(model.state_dict(), os.path.join(args.save_model_path, 'bestModel.pth'))

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, '
              'test_fscore: {}, time: {} sec'.format(e + 1, train_loss, train_acc, train_fscore,
                                                     test_loss, test_acc, test_fscore,
                                                     round(time.time() - start_time, 2)))

    print('Model Performance:')
    print('Best_Test-FScore-epoch_index: {}'.format(all_test_fscore.index(max(all_test_fscore)) + 1))
    print('Best_Test_F-Score: {}'.format(max(all_test_fscore)))
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0))
    confuPLT(confusion_matrix(best_label, best_pred, sample_weight=best_mask).astype(int), args.Dataset,
             save_path=os.path.join(args.save_model_path, 'heatmap.png'))

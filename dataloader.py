import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate_fn(data):
    # each item: (text, visual, audio, speaker, umask, label)
    text, visual, audio, speaker, umask, label = zip(*data)
    return [
        pad_sequence(text),                       # [T, B, D_t]
        pad_sequence(visual),                     # [T, B, D_v]
        pad_sequence(audio),                      # [T, B, D_a]
        pad_sequence(speaker),                    # [T, B, n_speakers]
        pad_sequence(umask, batch_first=True),    # [B, T]
        pad_sequence(label, batch_first=True),    # [B, T]
    ]


class IEMOCAPDataset(Dataset):
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]]),
            torch.FloatTensor([1] * len(self.videoLabels[vid])),
            torch.LongTensor(self.videoLabels[vid])
        )

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        return collate_fn(data)


class MELDDataset(Dataset):
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid, _ = pickle.load(open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor(np.array(self.videoSpeakers[vid])),
            torch.FloatTensor([1] * len(self.videoLabels[vid])),
            torch.LongTensor(self.videoLabels[vid])
        )

    def __len__(self):
        return self.len

    def return_labels(self):
        return [label for key in self.keys for label in self.videoLabels[key]]

    def collate_fn(self, data):
        return collate_fn(data)

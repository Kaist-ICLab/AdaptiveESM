import os
from os.path import join, expanduser

import numpy as np
import torch

from torch.utils.data import Dataset
from utils import get_user_esms


class DLBase(Dataset):
    """DailyLife base dataset class"""
    def __init__(self, dataset_path, esm_path, target, uids=None):
        self.dataset_path = expanduser(dataset_path)
        self.user_esms = get_user_esms(expanduser(esm_path))
        self.inputs, self.labels = self.load_data(target, uids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        inp = torch.Tensor(self.inputs[i])
        label = self.labels[i]
        return inp, label

    def get_label(self, uid, fname, target):
        esms = self.user_esms[uid]
        return esms.loc[int(fname.split('-')[0])][target]

    def load_data(self, target, uids):
        inputs, labels = [], []
        if uids is None:
            uids = self.user_esms.keys()

        for uid in uids:
            loadfrom = join(self.dataset_path, str(uid))
            inputs.extend([np.load(join(loadfrom, fname)) for fname in os.listdir(loadfrom)])
            labels.extend([self.get_label(uid, fname, target) for fname in os.listdir(loadfrom)])
        return inputs, labels


class WrapperDataset(Dataset):
    """Wrapper dataset class"""
    def __init__(self, inputs, labels):
        self.inputs, self.labels = inputs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        inp = torch.Tensor(self.inputs[i])
        label = self.labels[i]
        return inp, label

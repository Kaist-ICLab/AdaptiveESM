import os
import json
import numpy as np

from collections import OrderedDict, Counter
from scipy.interpolate import interp1d
from scipy.signal import decimate

import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, random_split, DataLoader


class KEMOCONDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size, label_type, n_classes, val_size, resample=False, standardize=False, fusion=None, label_fn=None):
        super().__init__()
        assert label_type in {'self', 'partner', 'external'}, f'label_type must be one of "self", "partner", and "external", but given "{label_type}".'
        # assert fusion is None or fusion in {'stack', 'decision', 'autoencoder'}, f'fusion must be one of "feature", "decision", and "autoencoder", but given "{fusion}".'

        self.data_dir       = os.path.expanduser(data_dir)
        self.batch_size     = batch_size
        self.label_type     = label_type
        self.n_classes      = n_classes
        self.val_size       = val_size

        self.resample       = resample
        self.standardize    = standardize
        self.fusion         = fusion
        self.label_fn       = label_fn
        self.ids            = self.get_ids()

    def prepare_data(self):
        # Note: prepare_data is called from a single GPU. Do not use it to assign state (self.x = y)
        # load data from data_dir
        pid_to_segments = {}

        # for each participant
        for pid in map(int, os.listdir(self.data_dir)):
            pid_to_segments.setdefault(pid, [])
            curr_x = pid_to_segments[pid]

            # for segments for a participant
            pdir = os.path.join(self.data_dir, str(pid))
            for segname in os.listdir(pdir):
                # get segment index
                idx = int(segname.split('-')[1])
                # get labels
                labels = segname.split('-')[-1].split('.')[0]
                if self.label_type == 'self':
                    a, v = int(labels[0]), int(labels[1])
                elif self.label_type == 'partner':
                    a, v = int(labels[2]), int(labels[3])
                elif self.label_type == 'external':
                    a, v = int(labels[4]), int(labels[5])

                # transform labels using label_fn if given
                if self.label_fn is not None:
                    label = self.label_fn(a, v)

                # get signals
                with open(os.path.join(pdir, segname)) as f:
                    seg = json.load(f)
                    # adjust signal lengths (cut or pad right edge)
                    for sigtype, sr in [('bvp', 64), ('eda', 4), ('temp', 4), ('ecg', 1)]:
                        curr_sig = seg[sigtype]
                        seg[sigtype] = curr_sig[:sr * 5] if len(curr_sig) > sr * 5 else np.pad(curr_sig, pad_width=(0, sr * 5 - len(curr_sig)), mode='edge')
                    curr_x.append([idx, seg, label])

            # check label distribution: if the number of unique classes for the current participant does not equal n_classes,
            # we will remove current participant from the dataset as such participant's data cannot be used as a test set 
            if len(Counter(map(lambda x: x[-1], curr_x))) != self.n_classes:
                del pid_to_segments[pid]
                continue
                
            # apply up/downsampling
            if self.resample:
                # for each segment
                for _, seg, _ in curr_x:
                    # upsample ecg signals from 1hz to 4hz
                    x = np.linspace(1, 5, num=5, endpoint=True)
                    x_new = np.linspace(1, 5, num=20, endpoint=True)
                    seg['ecg'] = interp1d(x, seg['ecg'], kind='quadratic')(x_new)
                    # downsample bvp signals from 64hz to 4hz
                    seg['bvp'] = decimate(seg['bvp'], 16, zero_phase=True)

            # apply standardization
            if self.standardize:
                # for each signal type
                for sigtype in ['bvp', 'eda', 'temp', 'ecg']:
                    # get all signal values of type to compute mean and std
                    sig = np.concatenate(list(map(lambda x: x[1][sigtype], curr_x)))
                    # standardize each segment with mean and std
                    for _, seg, _ in curr_x:
                        seg[sigtype] = (seg[sigtype] - np.mean(sig)) / np.std(sig)

            # apply feature fusion by stacking
            if self.resample and self.fusion == 'stack':
                for i, (_, seg, _) in enumerate(curr_x):
                    curr_x[i][1] = torch.Tensor([seg['bvp'], seg['eda'], seg['temp'], seg['ecg']]).T

            # sort segments for current participant by segment indices
            curr_x.sort(key=lambda x: x[0])

        # return dict sorted by pid
        return OrderedDict(sorted(pid_to_segments.items(), key=lambda x: x[0]))

    def get_ids(self):
        return self.prepare_data().keys()

    def setup(self, stage=None, test_id=1):
        # setup expects a string arg stage. It is used to separate setup logic for trainer.fit and trainer.test.
        # assign train/val split(s) for use in dataloaders
        data = self.prepare_data()
        
        # for loso cross-validation
        if test_id is not None:
            if stage == 'test' or stage is None:
                inp, tgt = zip(*[(seg, label) for _, seg, label in data[test_id]])
                self.kemocon_test = TensorDataset(torch.stack(inp), torch.Tensor(tgt).unsqueeze(1))
                self.dims = tuple(self.kemocon_test[0][0].shape)

            if stage == 'fit' or stage is None:
                inp, tgt = zip(*[(seg, label) for pid in data if pid != test_id for _, seg, label in data[pid]])
                kemocon_full = TensorDataset(torch.stack(inp), torch.Tensor(tgt).unsqueeze(1))
                n_val = int(self.val_size * len(kemocon_full))
                self.kemocon_train, self.kemocon_val = random_split(
                    dataset     = kemocon_full,
                    lengths     = [len(kemocon_full) - n_val, n_val],
                    generator   = torch.Generator(),
                )
                self.dims = tuple(self.kemocon_train[0][0].shape)
        
        # test id is None, we are doing standard train/valid/test split
        # given val_size which is a float between 0 and 1 defining the proportion of validation set
        # validation and test sets will have the same size of val_size * full dataset, and train set will be the rest of the data
        else:
            inp, tgt = zip(*[(seg, label) for pid in data for _, seg, label in data[pid]])
            kemocon_full = TensorDataset(torch.stack(inp), torch.Tensor(tgt).unsqueeze(1))
            n_val = int(self.val_size * len(kemocon_full))
            train, valid, test = random_split(
                dataset     = kemocon_full,
                lengths     = [len(kemocon_full) - (n_val * 2), n_val, n_val],
                generator   = torch.Generator(),
            )

            if stage == 'fit' or stage is None:
                self.kemocon_train, self.kemocon_val = train, valid
                self.dims = tuple(self.kemocon_train[0][0].shape)
            
            if stage == 'test' or stage is None:
                self.kemocon_test = test
                self.dims = tuple(self.kemocon_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.kemocon_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.kemocon_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.kemocon_test, batch_size=self.batch_size)

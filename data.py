import os
import json
import warnings
import numpy as np
from tqdm import tqdm

from scipy.signal import decimate
from scipy.interpolate import interp1d
from collections import OrderedDict, Counter
from sklearn.preprocessing import StandardScaler
from pyteap.signals import bvp, gsr, hst, ecg

import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, random_split, DataLoader


class KEMOCONDataModule(pl.LightningDataModule):
    
    # data shape: [B, 20, 4]
    def __init__(self, config, label_fn=None):
        super().__init__()
        assert config.label_type in {'self', 'partner', 'external'}, f'label_type must be one of "self", "partner", and "external", but given "{config.label_type}".'
        # assert fusion is None or fusion in {'stack', 'decision', 'autoencoder'}, f'fusion must be one of "feature", "decision", and "autoencoder", but given "{fusion}".'

        self.sigtypes           = ['bvp', 'eda', 'temp', 'ecg']
        self.sample_rates       = [64, 4, 4, 1]

        self.data_dir           = os.path.expanduser(config.data_dir)
        self.batch_size         = config.batch_size
        self.label_type         = config.label_type
        self.n_classes          = config.n_classes
        self.val_size           = config.val_size
        self.num_segs           = config.num_segs  # number of segments in one input (minimum 5)

        self.resample           = config.resample
        self.extract_features   = config.extract_features
        self.standardize        = config.standardize
        self.fusion             = config.fusion
        self.label_fn           = label_fn
        self.ids                = self.get_ids()

        if self.resample and self.extract_features:
            warnings.warn('Resampling and feature extraction are mutually exclusive (cannot extract features from downsampled BVP signals), extract_features is set to false.', UserWarning)
            self.extract_features = False

    def get_features(self, sig, sr, sigtype):
        if sigtype == 'bvp':
            features = bvp.get_bvp_features(bvp.acquire_bvp(sig, sr), sr)
        elif sigtype == 'eda':
            features = gsr.get_gsr_features(gsr.acquire_gsr(sig, sr, conversion=1e6), sr)
        elif sigtype == 'temp':
            features = hst.get_hst_features(hst.acquire_hst(sig, sr), sr)
        elif sigtype == 'ecg':
            features = ecg.get_ecg_features(sig)
        return features

    def prepare_data(self, check_id=False):
        # Note: prepare_data is called from a single GPU. Do not use it to assign state (self.x = y)
        # load data from data_dir
        pid_to_segments = dict()

        # for each participant
        for pid in sorted(map(int, os.listdir(self.data_dir))):
            pid_to_segments.setdefault(pid, list())
            segments = pid_to_segments[pid]

            # load segments
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
                else:
                    label = (a, v)

                # get signals (5s long)
                with open(os.path.join(pdir, segname)) as f:
                    seg = json.load(f)
                    # adjust signal lengths (cut or pad right edge)
                    for sigtype, sr in zip(self.sigtypes, self.sample_rates):
                        sig = seg[sigtype]
                        seg[sigtype] = sig[:sr * 5] if len(sig) > sr * 5 else np.pad(sig, pad_width=(0, sr * 5 - len(sig)), mode='edge')
                    # append current segment to list
                    segments.append([idx, seg, label])

            # sort list of segments by index
            segments.sort(key=lambda x: x[0])

            # check label distribution: if the number of unique classes for the current participant does not equal n_classes,
            # remove current participant from the dataset as such participant's data cannot be used for testing
            if len(Counter(map(lambda x: x[-1], segments))) != self.n_classes:
                del pid_to_segments[pid]
                continue
            
            # break loop if only checking for participant ids in the data
            if check_id:
                continue
            
            # concat N = num_segs segments (each 5s) via a rolling method
            curr_x = list()
            for i in range(len(segments) - self.num_segs + 1):
                segs = segments[i:i + self.num_segs]  # segments to be concatenated
                # concat segments for each signal type
                seg = {sigtype: np.concatenate([seg[sigtype] for _, seg, _ in segs]) for sigtype in self.sigtypes}
                curr_x.append([i, seg, segs[-1][-1]])  # take the label of the last segment

            # apply up/downsampling - upsample ECG (=heart rate) and downsample bvp
            if self.resample:
                # for each segment
                for i, (_, seg, _) in enumerate(curr_x):
                    # upsample ecg signals from 1hz to 4hz
                    x = np.linspace(1, (self.num_segs * 5), num=(self.num_segs * 5), endpoint=True)
                    x_new = np.linspace(1, (self.num_segs * 5), num=(self.num_segs * 5 * 4), endpoint=True)
                    seg['ecg'] = interp1d(x, seg['ecg'], kind='quadratic')(x_new)
                    # downsample bvp signals from 64hz to 4hz
                    seg['bvp'] = decimate(seg['bvp'], 16, zero_phase=True)

            # extract features: resampling and feature extraction are mutually exclusive (can't extract features from resampled BVP signals)
            if not self.resample and self.extract_features:
                idx_to_del = list()  # list to store indices that will be deleted
                for i, (idx, seg, label) in tqdm(enumerate(curr_x), desc=f'Participant {pid:02d}', ascii=True, dynamic_ncols=True):
                    features = np.concatenate([self.get_features(seg[sigtype], sr, sigtype) for sigtype, sr in zip(self.sigtypes, self.sample_rates)])
                    # replace raw signals with features
                    curr_x[i] = [idx, features, label]
                    # current item will be removed if there is any feature that is nan
                    if np.isnan(features).any():
                        idx_to_del.append(i)
                # delete items with index in idx_to_del
                if idx_to_del:
                    curr_x = np.delete(curr_x, idx_to_del, axis=0).tolist()

            # apply signal-wise standardization (to raw signals)
            if self.standardize and not self.extract_features:
                # for each signal type
                for sigtype in self.sigtypes:
                    # concat all signals of type along time axis to compute mean and std
                    sig = np.concatenate(list(map(lambda x: x[1][sigtype], curr_x)))
                    # standardize each segment with mean and std
                    for _, seg, _ in curr_x:
                        seg[sigtype] = (seg[sigtype] - np.mean(sig)) / np.std(sig)

            # apply feature-wise standardization (to extracted features)
            elif self.standardize and self.extract_features:
                scaler = StandardScaler()
                scaler.fit(np.stack([features for _, features, _ in curr_x]))  # fit scaler to features
                for i, (_, features, _) in enumerate(curr_x):
                    curr_x[i][1] = np.squeeze(scaler.transform(features.reshape(1, -1)))  # standardize features with scaler

            # apply fusion by stacking (only to raw signals)
            if self.resample and not self.extract_features and self.fusion == 'stack':
                for i, (_, seg, _) in enumerate(curr_x):
                    curr_x[i][1] = np.transpose([seg[sigtype] for sigtype in self.sigtypes])

            # save processed segments to dict
            pid_to_segments[pid] = curr_x

        # return dict sorted by pid
        return OrderedDict(sorted(pid_to_segments.items(), key=lambda x: x[0]))

    def get_ids(self):
        return self.prepare_data(check_id=True).keys()

    def setup(self, stage=None, test_id=None):
        # setup expects a string arg stage. It is used to separate setup logic for trainer.fit and trainer.test.
        # assign train/val split(s) for use in dataloaders
        data = self.prepare_data()
        self.size_ = sum(len(data[pid]) for pid in data)  # total number of samples in the dataset
        
        # for loso cross-validation
        if test_id is not None:
            if stage == 'test' or stage is None:
                inp, tgt = zip(*[(torch.Tensor(seg), label) for _, seg, label in data[test_id]])
                self.kemocon_test = TensorDataset(torch.stack(inp), torch.Tensor(tgt).unsqueeze(1))
                self.dims = tuple(self.kemocon_test[0][0].shape)

            if stage == 'fit' or stage is None:
                inp, tgt = zip(*[(torch.Tensor(seg), label) for pid in data if pid != test_id for _, seg, label in data[pid]])
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
            inp, tgt = zip(*[(torch.Tensor(seg), label) for pid in data for _, seg, label in data[pid]])
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

    def full_dataset(self):
        # returns the full dataset except the testset as a TensorDataset
        return TensorDataset(
            *map(lambda x: torch.cat(x, dim=0), zip(self.kemocon_train[:], self.kemocon_val[:]))
        )

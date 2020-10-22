import os
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from ..data import KEMOCONDataModule


class SimpleLSTM(pl.LightningModule):

    def __init__(self, inp_size, out_size, hidden_size, n_layers, learning_rate=1e-3, p_drop=0.0):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(inp_size, hidden_size, n_layers, dropout=p_drop, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out)[:, -1]
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        out, _ = self.lstm(x)
        logits = self.fc(out)[:, -1]  # if batch_first=True
        # see https://discuss.pytorch.org/t/should-it-really-be-necessary-to-do-var-detach-cpu-numpy/35489
        # and https://stackoverflow.com/questions/63582590/why-do-we-call-detach-before-calling-numpy-on-a-pytorch-tensor
        # for converting tensors to numpy for calculating metrics
        # https://pytorch-lightning.readthedocs.io/en/latest/performance.html
        probas = torch.sigmoid(logits).detach().cpu().numpy()
        _y = y.detach().cpu().numpy()

        # for the choice of metrics, see https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc
        loss = nn.BCEWithLogitsLoss()(logits, y)
        acc = accuracy_score(_y, probas > 0.5)
        ap = average_precision_score(_y, probas, average='weighted', pos_label=1)
        f1 = f1_score(_y, probas > 0.5, average='weighted', pos_label=1)
        auroc = roc_auc_score(_y, probas, average='weighted')

        # converting scalars to tensors to prevent errors
        # see https://github.com/PyTorchLightning/pytorch-lightning/issues/3276
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', torch.tensor(acc), on_step=False, on_epoch=True, logger=True)
        self.log('train_ap', torch.tensor(ap), on_step=False, on_epoch=True, logger=True)
        self.log('train_f1', torch.tensor(f1), on_step=False, on_epoch=True, logger=True)
        self.log('train_auroc', torch.tensor(auroc), on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out, _ = self.lstm(x)
        logits = self.fc(out)[:, -1]
        probas = torch.sigmoid(logits).detach().cpu().numpy()
        _y = y.detach().cpu().numpy()

        loss = nn.BCEWithLogitsLoss()(logits, y)
        acc = accuracy_score(_y, probas > 0.5)
        ap = average_precision_score(_y, probas, average='weighted', pos_label=1)
        f1 = f1_score(_y, probas > 0.5, average='weighted', pos_label=1)
        auroc = roc_auc_score(_y, probas, average='weighted')

        self.log('valid_loss', loss, logger=True)
        self.log('valid_acc', torch.tensor(acc), logger=True)
        self.log('valid_ap', torch.tensor(ap), logger=True)
        self.log('valid_f1', torch.tensor(f1), logger=True)
        self.log('valid_auroc', torch.tensor(auroc), logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out, _ = self.lstm(x)
        logits = self.fc(out)[:, -1]
        probas = torch.sigmoid(logits).detach().cpu().numpy()
        _y = y.detach().cpu().numpy()

        loss = nn.BCEWithLogitsLoss()(logits, y)
        acc = accuracy_score(_y, probas > 0.5)
        ap = average_precision_score(_y, probas, average='weighted', pos_label=1)
        f1 = f1_score(_y, probas > 0.5, average='weighted', pos_label=1)
        auroc = roc_auc_score(_y, probas, average='weighted')

        self.log('test_loss', loss, logger=True)
        self.log('test_acc', torch.tensor(acc), logger=True)
        self.log('test_ap', torch.tensor(ap), logger=True)
        self.log('test_f1', torch.tensor(f1), logger=True)
        self.log('test_auroc', torch.tensor(auroc), logger=True)
        return {'loss': loss, 'acc': acc, 'ap': ap, 'f1': f1, 'auroc': auroc}

    def test_step_end(self, outputs):
        self.results = outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

import os
import numpy as np

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, confusion_matrix


class LSTM(pl.LightningModule):

    def __init__(self, inp_size, out_size, hidden_size, n_layers, p_drop=0.0, bidirectional=False, learning_rate=1e-3, name='default'):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(inp_size, hidden_size, n_layers, dropout=p_drop, batch_first=True, bidirectional=bidirectional)

        if bidirectional is True:
            self.fc = nn.Linear(hidden_size * 2, out_size)
        else:
            self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out)[:, -1]
        return logits

    def log_metrics(self, loss, y, probas, stage):
        # for the choice of metrics, see https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc
        acc = accuracy_score(y, probas > 0.5)
        ap = average_precision_score(y, probas, average='weighted', pos_label=1)
        f1 = f1_score(y, probas > 0.5, average='weighted', pos_label=1)
        auroc = roc_auc_score(y, probas, average='weighted')

        # converting scalars to tensors to prevent errors
        # see https://github.com/PyTorchLightning/pytorch-lightning/issues/3276
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log(f'{stage}_acc', torch.tensor(acc), on_step=False, on_epoch=True, logger=True)
        self.log(f'{stage}_ap', torch.tensor(ap), on_step=False, on_epoch=True, logger=True)
        self.log(f'{stage}_f1', torch.tensor(f1), on_step=False, on_epoch=True, logger=True)
        self.log(f'{stage}_auroc', torch.tensor(auroc), on_step=False, on_epoch=True, logger=True)

        if stage == 'test':
            return acc, ap, f1, auroc

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

        loss = nn.BCEWithLogitsLoss()(logits, y)
        self.log_metrics(loss, _y, probas, stage='train')

        # self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        # self.log('train_acc', torch.tensor(acc), on_step=False, on_epoch=True, logger=True)
        # self.log('train_ap', torch.tensor(ap), on_step=False, on_epoch=True, logger=True)
        # self.log('train_f1', torch.tensor(f1), on_step=False, on_epoch=True, logger=True)
        # self.log('train_auroc', torch.tensor(auroc), on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out, _ = self.lstm(x)
        logits = self.fc(out)[:, -1]
        probas = torch.sigmoid(logits).detach().cpu().numpy()
        _y = y.detach().cpu().numpy()

        loss = nn.BCEWithLogitsLoss()(logits, y)
        self.log_metrics(loss, _y, probas, stage='valid')
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out, _ = self.lstm(x)
        logits = self.fc(out)[:, -1]
        probas = torch.sigmoid(logits).detach().cpu().numpy()
        _y = y.detach().cpu().numpy()

        loss = nn.BCEWithLogitsLoss()(logits, y)
        acc, ap, f1, auroc = self.log_metrics(loss, _y, probas, stage='test')
        cm = confusion_matrix(_y, probas > 0.5, normalize=None)
        return {'acc': acc, 'ap': ap, 'f1': f1, 'auroc': auroc}, cm

    def test_step_end(self, outputs):
        self.results = outputs

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1)
        return [optimizer], [scheduler]


class AttentionLSTM(pl.LightningModule):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def test_step_end(self, outputs):
        pass

    def configure_optimizers(self, outputs):
        pass
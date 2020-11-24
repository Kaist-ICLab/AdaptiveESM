import os
import numpy as np
import xgboost as xgb
from xgboost import DMatrix

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data

from torch import nn
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, confusion_matrix
from utils import get_config


class LSTM(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        # save hyperparameters, anything assigned to self.hparams will be saved automatically
        self.hparams = hparams

        # define LSTM and fully-connected layer
        self.lstm = nn.LSTM(
            input_size      = hparams.inp_size,
            hidden_size     = hparams.hidden_size,
            num_layers      = hparams.n_layers,
            dropout         = hparams.p_drop,
            bidirectional   = hparams.bidirectional,
            batch_first     = True
        )
        if hparams.bidirectional is True:
            self.fc = nn.Linear(hparams.hidden_size * 2, hparams.out_size)
        else:
            self.fc = nn.Linear(hparams.hidden_size, hparams.out_size)
        
        # define loss
        self.loss = nn.BCEWithLogitsLoss()

    @auto_move_data
    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out)[:, -1]  # if batch_first=True
        return logits

    def log_metrics(self, loss, logits, y, stage):
        # see https://discuss.pytorch.org/t/should-it-really-be-necessary-to-do-var-detach-cpu-numpy/35489
        # and https://stackoverflow.com/questions/63582590/why-do-we-call-detach-before-calling-numpy-on-a-pytorch-tensor
        # for converting tensors to numpy for calculating metrics
        # https://pytorch-lightning.readthedocs.io/en/latest/performance.html
        probas = torch.sigmoid(logits).detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        # for the choice of metrics, see https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc
        acc = accuracy_score(y, probas > 0.5)
        ap = average_precision_score(y, probas, average='weighted', pos_label=1)
        f1 = f1_score(y, probas > 0.5, average='weighted', pos_label=1)
        auroc = roc_auc_score(y, probas, average='weighted')

        # converting scalars to tensors to prevent errors
        # see https://github.com/PyTorchLightning/pytorch-lightning/issues/3276
        self.log_dict({
            f'{stage}_loss': loss,
            f'{stage}_acc': torch.tensor(acc),
            f'{stage}_ap': torch.tensor(ap),
            f'{stage}_f1': torch.tensor(f1),
            f'{stage}_auroc': torch.tensor(auroc)
        }, on_step=False, on_epoch=True, logger=True)

        if stage == 'test':
            cm = confusion_matrix(y, probas > 0.5, normalize=None)
            return cm

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log_metrics(loss, logits, y, stage='train')
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log_metrics(loss, logits, y, stage='valid')
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        cm = self.log_metrics(loss, logits, y, stage='test')
        return cm

    def test_epoch_end(self, outputs):
        # save test confusion matrix
        self.cm = sum(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        # configure learning rate scheduler if needed
        if self.hparams.scheduler is not None:
            if self.hparams.scheduler.type == 'CosineAnnealingWarmRestarts':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **vars(self.hparams.scheduler.params))
                return [optimizer], [scheduler]

            elif self.hparams.scheduler.type == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **vars(self.hparams.scheduler.params))
                return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_loss'}

        else:
            return optimizer


class StackedLSTM(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lstm1 = nn.LSTM(4, 50, 1, bias=True, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(100, 50, 1, bias=True, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(100, 1, bias=True)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(self.dropout(out))
        logits = self.fc(self.dropout(out))[:, -1]
        return logits

    def log_metrics(self, loss, logits, y, stage):
        probas = torch.sigmoid(logits).detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        acc = accuracy_score(y, probas > 0.5)
        ap = average_precision_score(y, probas, average='weighted', pos_label=1)
        f1 = f1_score(y, probas > 0.5, average='weighted', pos_label=1)
        auroc = roc_auc_score(y, probas, average='weighted')

        self.log_dict({
            f'{stage}_loss': loss,
            f'{stage}_acc': torch.tensor(acc),
            f'{stage}_ap': torch.tensor(ap),
            f'{stage}_f1': torch.tensor(f1),
            f'{stage}_auroc': torch.tensor(auroc)
        }, on_step=False, on_epoch=True, logger=True)

        if stage == 'test':
            cm = confusion_matrix(y, probas > 0.5, normalize=None)
            return cm

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log_metrics(loss, logits, y, stage='train')
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log_metrics(loss, logits, y, stage='valid')
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        cm = self.log_metrics(loss, logits, y, stage='test')
        return cm

    def test_epoch_end(self, outputs):
        self.test_confmat = sum(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerNet(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # positional encoding
        self.pe = PositionalEncoding(d_model=hparams.d_model, max_len=hparams.max_len)

        # encoder layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = hparams.d_model,
            nhead           = hparams.nhead,
            dim_feedforward = hparams.dim_feedforward,
            dropout         = hparams.dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=hparams.num_layers)

        # dense layer
        self.dense = nn.Linear(hparams.d_model * hparams.max_len, hparams.out_size)

        # loss function
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # swap batch and seq_len dimension if batch comes first
        if self.hparams.batch_first:
            x = torch.transpose(x, 0, 1)
        
        x = self.pe(x)
        x = self.encoder(x)
        x = x.reshape(x.shape[1], -1)
        logits = self.dense(x)
        return logits

    def log_metrics(self, loss, logits, y, stage):
        probas = torch.sigmoid(logits).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        
        acc = accuracy_score(y, probas > 0.5)
        ap = average_precision_score(y, probas, average='weighted', pos_label=1)
        f1 = f1_score(y, probas > 0.5, average='weighted', pos_label=1)
        auroc = roc_auc_score(y, probas, average='weighted')

        self.log_dict({
            f'{stage}_loss': loss,
            f'{stage}_acc': torch.tensor(acc),
            f'{stage}_ap': torch.tensor(ap),
            f'{stage}_f1': torch.tensor(f1),
            f'{stage}_auroc': torch.tensor(auroc)
        }, on_step=False, on_epoch=True, logger=True)

        if stage == 'test':
            cm = confusion_matrix(y, probas > 0.5, normalize=None)
            return cm

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log_metrics(loss, logits, y, stage='train')
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log_metrics(loss, logits, y, stage='valid')
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        cm = self.log_metrics(loss, logits, y, stage='test')
        return cm

    def test_epoch_end(self, outputs):
        self.test_confmat = sum(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    
class XGBoost(object):

    def __init__(self, hparams):
        self.hparams = hparams

    def train(self, x, y):
        self.dtrain = DMatrix(x, label=y)
        self.bst = xgb.train(vars(self.hparams.bst), self.dtrain, self.hparams.num_rounds)

    def predict(self, x):
        probs = self.bst.predict(DMatrix(x))
        preds = probs > self.hparams.threshold
        return probs, preds

    def test(self, x, y):
        probs, preds = self.predict(x)

        # get metrics
        acc = accuracy_score(y, preds)
        ap = average_precision_score(y, probs, average='weighted', pos_label=1)
        f1 = f1_score(y, preds, average='weighted', pos_label=1)
        auroc = roc_auc_score(y, probs, average='weighted')
        cm = confusion_matrix(y, preds, normalize=None)

        return {'acc': acc, 'ap': ap, 'f1': f1, 'auroc': auroc}, cm

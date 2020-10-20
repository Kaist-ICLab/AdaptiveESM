import os
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.metrics import balanced_accuracy_score, average_precision_score, f1_score
from data import KEMOCONDataModule


class SimpleLSTM(pl.LightningModule):

    def __init__(self, inp_size, out_size, hidden_size, n_layers, learning_rate=1e-3, p=0.0):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(inp_size, hidden_size, n_layers, dropout=p, batch_first=True)
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
        probas = torch.sigmoid(logits).detach().cpu().numpy()
        _y = y.detach().cpu().numpy()

        # for the choice of metrics, see https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc
        loss = nn.BCEWithLogitsLoss()(logits, y)
        acc = balanced_accuracy_score(_y, probas > 0.5)
        ap = average_precision_score(_y, probas, average='weighted', pos_label=1)
        f1 = f1_score(_y, probas > 0.5, average='weighted', pos_label=1)

        # converting scalars to tensors to prevent errors
        # see https://github.com/PyTorchLightning/pytorch-lightning/issues/3276
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', torch.tensor(acc), on_step=False, on_epoch=True, logger=True)
        self.log('train_ap', torch.tensor(ap), on_step=False, on_epoch=True, logger=True)
        self.log('train_f1', torch.tensor(f1), on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out, _ = self.lstm(x)
        logits = self.fc(out)[:, -1]
        probas = torch.sigmoid(logits).detach().cpu().numpy()
        _y = y.detach().cpu().numpy()

        loss = nn.BCEWithLogitsLoss()(logits, y)
        acc = balanced_accuracy_score(_y, probas > 0.5)
        ap = average_precision_score(_y, probas, average='weighted', pos_label=1)
        f1 = f1_score(_y, probas > 0.5, average='weighted', pos_label=1)

        self.log('valid_loss', loss, logger=True)
        self.log('valid_acc', torch.tensor(acc), logger=True)
        self.log('valid_ap', torch.tensor(ap), logger=True)
        self.log('valid_f1', torch.tensor(f1), logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out, _ = self.lstm(x)
        logits = self.fc(out)[:, -1]
        probas = torch.sigmoid(logits).detach().cpu().numpy()
        _y = y.detach().cpu().numpy()

        loss = nn.BCEWithLogitsLoss()(logits, y)
        acc = balanced_accuracy_score(_y, probas > 0.5)
        ap = average_precision_score(_y, probas, average='weighted', pos_label=1)
        f1 = f1_score(_y, probas > 0.5, average='weighted', pos_label=1)

        self.log('test_loss', loss, logger=True)
        self.log('test_acc', torch.tensor(acc), logger=True)
        self.log('test_ap', torch.tensor(ap), logger=True)
        self.log('test_f1', torch.tensor(f1), logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


if __name__ == "__main__":
    seed_everything(1)

    logger = TensorBoardLogger(
        save_dir    = 'logs',
        name        = 'simple_lstm',
    )

    early_stop_callback = EarlyStopping(
        monitor     = 'valid_loss',
        min_delta   = 0.0,
        patience    = 70,
        verbose     = True,
        mode        = 'min'
    )

    trainer = pl.Trainer(
        gpus                = 1,
        auto_select_gpus    = True,
        precision           = 16,
        logger              = logger,
        callbacks           = [early_stop_callback],
        auto_lr_find        = True,
        gradient_clip_val   = 0.5,
        deterministic       = True,
    )

    lstm = SimpleLSTM(
        inp_size        = 4,
        out_size        = 1,
        hidden_size     = 100,
        n_layers        = 2,
        p               = 0.3,
    )
    
    def arousal_binary(a, v):
        return int(a <= 2)

    def valence_binary(a, v):
        return int(v <= 2)

    dm = KEMOCONDataModule(
        data_dir    = '~/data/kemocon/segments',
        batch_size  = 2000,
        label_type  = 'self',
        n_classes   = 2,
        test_id     = 1,
        val_size    = 0.1,
        resample    = True,
        standardize = True,
        fusion      = 'stack',
        label_fn    = valence_binary,
    )

    trainer.tune(lstm, datamodule=dm)
    trainer.fit(lstm, dm)
    trainer.test(lstm)

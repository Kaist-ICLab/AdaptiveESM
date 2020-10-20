import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from data import KEMOCONDataModule


class SimpleLSTM(pl.LightningModule):

    def __init__(self, inp_size, out_size, hidden_size, n_layers, p=0.0):
        super().__init__()
        self.lstm = nn.LSTM(inp_size, hidden_size, n_layers, dropout=p, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # logits = self.fc(out)[-1]
        logits = self.fc(out)[:, -1]  # if batch_first=True
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        out, _ = self.lstm(x)
        # logits = self.fc(out)[-1]
        logits = self.fc(out)[:, -1]  # if batch_first=True

        loss = nn.BCEWithLogitsLoss()(logits, y.unsqueeze(1))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out, _ = self.lstm(x)
        # logits = self.fc(out)[-1]
        logits = self.fc(out)[:, -1]  # if batch_first=True

        val_loss = nn.BCEWithLogitsLoss()(logits, y.unsqueeze(1))
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    lstm = SimpleLSTM(
        inp_size    = 4,
        out_size    = 1,
        hidden_size = 100,
        n_layers    = 2,
    )

    trainer = pl.Trainer(
        gpus                = 1,
        auto_select_gpus    = True,
        precision           = 16,
    )
    
    def arousal_binary(a, v):
        return int(a > 2)

    def valence_binary(a, v):
        return int(v > 2)

    dm = KEMOCONDataModule(
        seed        = 1,
        data_dir    = '~/data/kemocon/segments',
        batch_size  = 100,
        label_type  = 'self',
        n_classes   = 2,
        test_id     = 1,
        val_size    = 0.1,
        resample    = True,
        standardize = True,
        fusion      = 'stack',
        label_fn    = valence_binary,
    )

    # dm.setup('fit')
    # print(dm.dims)
    trainer.fit(lstm, dm)

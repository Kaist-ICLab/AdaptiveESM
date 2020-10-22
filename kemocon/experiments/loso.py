import os
import argparse
import pandas as pd
from ..data import KEMOCONDataModule
from ..models.rnn import SimpleLSTM

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def transform_label(target, pos_label):

    def transform_fn(a, v):
        label = a if target == 'arousal' else v
        return int(label > 2) if pos_label == 'high' else int(label <= 2)

    return transform_fn


def run_loso_eval(seed, target, pos_label):
    # set seed, see: https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#reproducibility
    seed_everything(seed)
    results = []

    dm = KEMOCONDataModule(
        data_dir    = '~/data/kemocon/segments',
        batch_size  = 2000,
        label_type  = 'self',
        n_classes   = 2,
        val_size    = 0.1,
        resample    = True,
        standardize = True,
        fusion      = 'stack',
        label_fn    = transform_label(target, pos_label),
    )
    
    # for each participant in datamodule
    for pid in dm.ids:
        # make logger
        logger = TensorBoardLogger(
            save_dir    = os.path.expanduser('~/projects/AdaptiveESM/logs'),
            name        = f'loso_{target}_{pos_label}',
            version     = f'{pid:02d}'
        )
        # define early stopping, see: https://pytorch-lightning.readthedocs.io/en/latest/early_stopping.html
        early_stop_callback = EarlyStopping(
            monitor     = 'valid_loss',
            min_delta   = 0.0,
            patience    = 20,
            verbose     = True,
            mode        = 'min'
        )
        # make trainer
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
        # make model
        model = SimpleLSTM(
            inp_size        = 4,
            out_size        = 1,
            hidden_size     = 100,
            n_layers        = 2,
            p_drop          = 0.3,
        )

        # find optimal LR, see: https://pytorch-lightning.readthedocs.io/en/latest/lr_finder.html#learning-rate-finder
        trainer.tune(model, datamodule=dm)

        # train model
        dm.setup(stage='fit', test_id=pid)
        trainer.fit(model, dm)

        # test model
        dm.setup(stage='test', test_id=pid)
        trainer.test(model)
        
        # get results
        res = model.results
        res['num_epochs'] = model.current_epoch
        res['lr'] = model.hparams.learning_rate
        res['pid'] = pid
        results.append(res)

    results = pd.DataFrame(results).set_index('pid')
    results.to_csv(os.path.expanduser(f'~/projects/AdaptiveESM/results/loso_{target}_{pos_label}.csv'))


if __name__ == "__main__":
    # init parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default='arousal')
    parser.add_argument('-p', '--pos', type=str, default='high')
    args = parser.parse_args()

    run_loso_eval(1, args.target, args.pos)

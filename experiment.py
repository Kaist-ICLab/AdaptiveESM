import os
import json
import argparse
import pandas as pd
from ..data import KEMOCONDataModule
from ..models.rnn import LSTM

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def transform_label(target, pos_label):

    def transform_fn(a, v):
        label = a if target == 'arousal' else v
        return int(label > 2) if pos_label == 'high' else int(label <= 2)

    return transform_fn


def parse_config(config):
    


def run_eval(config):
    # set seed, see: https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#reproducibility
    seed_everything(seed)

    # prepare data
    dm = KEMOCONDataModule(
        data_dir    = '~/data/kemocon/segments',
        batch_size  = 200,
        label_type  = 'self',
        n_classes   = 2,
        val_size    = 0.2,
        resample    = True,
        standardize = True,
        fusion      = 'stack',
        label_fn    = transform_label(target, pos_label),
    )

    # print participant ids in the current datamodule
    print(f'Running evaluation with data from participants: {dm.ids}')

    if eval_type == 'kfold':
        # make logger
        logger = TensorBoardLogger(
            save_dir    = os.path.expanduser('~/projects/AdaptiveESM/logs'),
            name        = f'kfold_{name}_{target}_{pos_label}',
        )
        # init LR monitor
        lr_


    if eval_type == 'loso':
        results, cms = list(), dict()

        # for each participant in datamodule
        for pid in dm.ids:
            # make logger
            logger = TensorBoardLogger(
                save_dir    = os.path.expanduser('~/projects/AdaptiveESM/logs'),
                name        = f'loso_{name}_{target}_{pos_label}',
                version     = f'{pid:02d}'
            )
            # init LR monitor
            lr_monitor = LearningRateMonitor(logging_interval='epoch')
            # define early stopping, see: https://pytorch-lightning.readthedocs.io/en/latest/early_stopping.html
            early_stop_callback = EarlyStopping(
                monitor     = 'valid_loss',
                min_delta   = 0.0,
                patience    = 150,
                verbose     = True,
                mode        = 'min'
            )
            # make trainer
            trainer = pl.Trainer(
                gpus                = 1,
                max_epochs          = 500,
                auto_select_gpus    = True,
                precision           = 16,
                logger              = logger,
                callbacks           = [lr_monitor, early_stop_callback],
                auto_lr_find        = True,
                gradient_clip_val   = 0.2,
                deterministic       = True,
            )
            # make model
            model = LSTM(
                inp_size        = 4,
                out_size        = 1,
                hidden_size     = 128,
                n_layers        = 2,
                p_drop          = 0.2,
                bidirectional   = True,
                name            = name,
                learning_rate   = 0.1,
            )

            # find optimal LR, see: https://pytorch-lightning.readthedocs.io/en/latest/lr_finder.html#learning-rate-finder
            trainer.tune(model, datamodule=dm)

            # train model
            dm.setup(stage='fit', test_id=pid)
            trainer.fit(model, dm)

            # test model
            dm.setup(stage='test', test_id=pid)
            trainer.test(model)
            
            # get metrics and confusion matrix
            metrics, cm = model.results
            metrics['num_epochs'] = model.current_epoch
            metrics['lr'] = model.hparams.learning_rate
            metrics['pid'] = pid
            results.append(metrics)

            # save confusion matrix
            cms[pid] = cm.tolist()
            print(f'### Confusion matrix for participant {pid} ###')
            print(cm)

        # save metrics as csv
        pd.DataFrame(results).set_index('pid').to_csv(os.path.expanduser(f'~/projects/AdaptiveESM/results/loso/{name}_{target}_{pos_label}_metrics.csv'))

        # pickle confusion matrices
        with open(os.path.expanduser(f'~/projects/AdaptiveESM/results/loso/{name}_{target}_{pos_label}_confmat.json'), 'w') as f:
            json.dump(cms, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    # init parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-n', '--name', type=str, default='default')
    parser.add_argument('-t', '--target', type=str, default='arousal')
    parser.add_argument('-p', '--pos', type=str, default='high')
    args = parser.parse_args()

    run_loso_eval(args.seed, args.name, args.target, args.pos)

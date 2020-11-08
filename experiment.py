# import system 
import os
import json
import argparse
import pandas as pd
from typing import Dict
from argparse import Namespace

# import custom modules
from data import KEMOCONDataModule
from utils import get_config, transform_label
from models import LSTM, StackedLSTM, TransformerNet

# import pytorch lightning related stuff
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Experiment(object):
    
    def __init__(self, config: str) -> None:
        # get configurations
        self.config = get_config(config)
        
        # set seed
        seed_everything(self.config.exp.seed)

        # prepare data
        self.dm = KEMOCONDataModule(
            config      = self.config.data,
            label_fn    = transform_label(self.config.exp.target, self.config.exp.pos_label),
        )

        # get experiment name
        exp_name = f'{self.config.exp.model}_{self.config.exp.type}_{self.config.exp.target}_{self.config.exp.pos_label}'
        print(f'Experiment: {exp_name}, w/ participants: {list(self.dm.ids)}')

        # make directory to save experiment results
        self.savedir = os.path.expanduser(os.path.join(self.config.exp.savedir, exp_name))
        os.makedirs(self.savedir, exist_ok=True)

    def get_logger(self, pid):
        # set version number if needed
        version = "" if pid is None else f'_{pid:02d}'

        # make logger
        if self.config.logger.type == 'tensorboard':
            logger = TensorBoardLogger(
                save_dir    = os.path.expanduser(self.config.logger.logdir),
                name        = f'{self.config.exp.model}_{self.config.exp.type}_{self.config.exp.target}_{self.config.exp.pos_label}{version}',
            )
        
        elif self.config.logger.type == 'comet':
            logger = CometLogger(
                api_key         = os.environ.get(self.config.logger.api_key),
                workspace       = os.environ.get(self.config.logger.workspace),
                save_dir        = os.path.expanduser(self.config.logger.logdir),
                project_name    = self.config.logger.project_name,
                experiment_name = f'{self.config.exp.model}_{self.config.exp.type}_{self.config.exp.target}_{self.config.exp.pos_label}{version}'
            )

        return logger

    def _body(self, pid=None):
        # get logger
        self.logger = self.get_logger(pid)

        # init lr monitor and callbacks
        self.callbacks = list()
        if self.config.hparams.scheduler is not None:
            self.callbacks.append(LearningRateMonitor(logging_interval='epoch'))

        # init early stopping
        if self.config.early_stop is not None:
            self.callbacks.append(EarlyStopping(**vars(self.config.early_stop)))

        # make model
        if self.config.exp.model == 'lstm':
            self.model = LSTM(self.config.hparams)
        elif self.config.exp.model == 'stacked_lstm':
            self.model = StackedLSTM(self.config.hparams)
        elif self.config.exp.model == 'transformer':
            self.model = TransformerNet(self.config.hparams)

        # make trainer
        trainer_args = vars(self.config.trainer)
        trainer_args.update({
            'logger': self.logger,
            'callbacks': self.callbacks,
            'auto_lr_find': True if self.config.exp.tune else False
        })
        self.trainer = pl.Trainer(**trainer_args)

        # find optimal lr
        if self.config.exp.tune:
            self.trainer.tune(self.model, datamodule=self.dm)
        
        # train model
        self.dm.setup(stage='fit', test_id=pid)
        self.trainer.fit(self.model, self.dm)

        # test model and get results
        self.dm.setup(stage='test', test_id=pid)
        results = self.trainer.test(self.model)[0]

        # return metrics and confusion matrix
        metrics = {
            'pid': pid,
            'acc': results['test_acc'],
            'ap': results['test_ap'],
            'f1': results['test_f1'],
            'auroc': results['test_auroc'],
            'num_epochs': self.model.current_epoch,
        }
        return metrics, self.model.test_confmat

    def run(self) -> None:
        # run k-fold cv
        if self.config.exp.type == 'kfold':
            metrics, cm = self._body()
            print(cm)

            # save results
            with open(os.path.join(self.savedir, 'results.json'), 'w') as f:
                json.dump({'metrics': metrics, 'confmat': cm.tolist()}, f, indent=4)

        if self.config.exp.type == 'loso':
            results, confmats = list(), dict()

            # for each participant
            for pid in self.dm.ids:
                # run loso cv and get results
                metrics, cm = self._body(pid=pid)
                results.append(metrics)
                confmats[pid] = cm.tolist()
                print(cm)

            # save metrics as csv
            pd.DataFrame(results).set_index('pid').to_csv(os.path.join(self.savedir, 'metrics.csv'))

            # save confusion matrices
            with open(os.path.join(self.savedir, 'confmat.json'), 'w') as f:
                json.dump(confmats, f, sort_keys=True, inden=4)


if __name__ == "__main__":
    # init parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to a configuration file for running an experiment')
    args = parser.parse_args()

    # run experiment with configuration
    exp = Experiment(args.config)
    exp.run()

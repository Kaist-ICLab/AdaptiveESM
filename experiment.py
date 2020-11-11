# import system 
import os
import time
import json
import argparse
import pandas as pd
from typing import Dict
from argparse import Namespace

# import custom modules
from data import KEMOCONDataModule
from utils import get_config, transform_label, config_to_dict
from models import LSTM, StackedLSTM, TransformerNet, XGBoost

# import pytorch related
import torch
from torch.utils.data import ConcatDataset

# and pytorch-lightning related stuff
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

        # make directory to save results
        os.makedirs(os.path.expanduser(self.config.exp.savedir), exist_ok=True)

        # set path to save experiment results
        self.savepath = os.path.expanduser(os.path.join(self.config.exp.savedir, f'{exp_name}_{int(time.time())}.json'))
        
    def init_logger(self, pid):
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

    def init_model(self, hparams):
        if self.config.exp.model == 'xgboost':
            model = XGBoost(hparams)
        elif self.config.exp.model == 'lstm':
            model = LSTM(hparams)
        elif self.config.exp.model == 'transformer':
            model = TransformerNet(hparams)
        return model

    def _body(self, pid=None):
        # make model
        self.model = self.init_model(self.config.hparams)

        # setup datamodule
        self.dm.setup(stage=None, test_id=pid)

        # init training for pl.LightningModule models
        if self.config.trainer is not None:
            # init logger
            if self.config.logger is not None:
                logger = self.init_logger(pid)

            # init lr monitor and callbacks
            callbacks = list()
            if self.config.hparams.scheduler is not None:
                callbacks.append(LearningRateMonitor(logging_interval='epoch'))

            # init early stopping
            if self.config.early_stop is not None:
                callbacks.append(EarlyStopping(**vars(self.config.early_stop)))

            trainer_args = vars(self.config.trainer)
            trainer_args.update({
                'logger': logger,
                'callbacks': callbacks,
                'auto_lr_find': True if self.config.exp.tune else False
            })
            trainer = pl.Trainer(**trainer_args)

            # find optimal lr
            if self.config.exp.tune:
                trainer.tune(self.model, datamodule=self.dm)

            # train model
            trainer.fit(self.model, self.dm)

            # test model and get results
            results = trainer.test(self.model)[0]

            # return metrics and confusion matrix
            metr = {
                'pid': pid,
                'acc': results['test_acc'],
                'ap': results['test_ap'],
                'f1': results['test_f1'],
                'auroc': results['test_auroc'],
                'num_epochs': self.model.current_epoch,
            }
            cm = self.model.test_confmat
        
        else:
            # train model: concat train and valid inputs and labels and convert torch tensors to numpy arrays
            X_train, y_train = map(lambda x: torch.cat(x, dim=0).numpy(), zip(self.dm.kemocon_train[:], self.dm.kemocon_val[:]))
            self.model.train(X_train, y_train)

            # test model
            X_test, y_test = map(lambda x: x.numpy(), self.dm.kemocon_test[:])
            metr, cm = self.model.test(X_test, y_test)

        return metr, cm

    # def _active_body(self, pid=None):
    #     self.model = self.init_model(self.config.hparams)

    def run(self) -> None:
        # run k-fold cv
        if self.config.exp.type == 'kfold':
            metr, cm = self._body()
            results = {
                'config': config_to_dict(self.config),
                'metrics': metr,
                'confmats': cm.tolist()
            }
            print(metr)
            print(cm)

        # run loso cv
        if self.config.exp.type == 'loso':
            metrics, confmats = list(), dict()

            # for each participant
            for pid in self.dm.ids:
                # run loso cv and get results
                metr, cm = self._body(pid=pid)
                metrics.append(metr)
                confmats[pid] = cm.tolist()
                print(f'pid: {pid},\n{cm}')

            # convert metrics for each participant to json string
            metrics = pd.DataFrame(metrics).set_index('pid').to_json(orient='index')

            # make results dict
            results = {
                'config': config_to_dict(self.config),
                'metrics': metrics,
                'confmats': confmats
            }

        # run active learning
        # if self.config.exp.type == 'active':
        #     metrics, cm = self._active_body()
        #     print(cm)

        # save results
        with open(self.savepath, 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    # init parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to a configuration file for running an experiment')
    args = parser.parse_args()

    # run experiment with configuration
    exp = Experiment(args.config)
    exp.run()

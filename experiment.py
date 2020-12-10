# import system 
import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict
from argparse import Namespace
from collections import Counter

# import xgboost
import xgboost as xgb
from xgboost import DMatrix

# import custom modules
from data import KEMOCONDataModule
from utils import get_config, transform_label, config_to_dict
from models import LSTM, StackedLSTM, TransformerNet, XGBoost

# import pytorch related
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader

# and pytorch-lightning related stuff
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class ActiveLearner(object):
    
    def __init__(self, config, model, datamodule):
        self.init_ratio             = config.init_ratio
        self.val_ratio              = config.val_ratio
        self.update_ratio           = config.update_ratio

        self.decision_boundary      = model.hparams.threshold if isinstance(model, XGBoost) else config.decision_boundary
        self.gamma                  = config.gamma
        self.alpha                  = config.alpha
        self.beta                   = config.beta

        self.update_lr              = config.update_lr
        self.update_epochs          = config.update_epochs

        self.model = model
        self.dm = datamodule
        self.queried = list()

    def _get_counts(self, targets=None):
        if targets is None:
            return Counter(torch.flatten(self.train_tgt).tolist())
        else:
            return Counter(torch.flatten(targets).tolist())

    def _get_distribution(self, targets):
        counts = self._get_counts(targets)
        return f'{counts[0]}({counts[0] / len(targets):.2f}):{counts[1]}({counts[1] / len(targets):.2f})'

    def initial_setup_info(self):
        info_msg = (
            "========== Initial setup ==========\n"
            f"Initial: {self._get_distribution(self.train_tgt)}\n"
            f"Validation: {self._get_distribution(self.valid_tgt) if len(self.valid_tgt) > 0 else None}\n"
            "==================================="
        )
        return info_msg

    def setup(self, pid):
        # setup datamodule and get full dataset
        self.dm.setup(stage=None, test_id=pid)
        full_data = self.dm.trainval_dataset()  # train + valid

        # split full dataset into initial batch and datastream
        self.full_size = len(full_data)
        init_size = round(self.full_size * self.init_ratio * (1 - self.val_ratio))
        val_size = round(self.full_size * self.init_ratio * self.val_ratio)

        self.init_batch, self.val_batch, self.datastream = random_split(
            dataset     = full_data,
            lengths     = [init_size, val_size, self.full_size - init_size - val_size],
            generator   = torch.Generator()
        )
        self.train_inp, self.train_tgt = self.init_batch[:]
        self.valid_inp, self.valid_tgt = self.val_batch[:]
        print(self.initial_setup_info())

        # make init loader and stream loader
        self.init_loader = DataLoader(self.init_batch, batch_size=init_size)
        self.val_loader = DataLoader(self.val_batch, batch_size=val_size) if val_size > 0 else None
        self.stream_loader = DataLoader(self.datastream, batch_size=1)

        # define update size and minority label
        self.update_size = round(self.full_size * self.update_ratio)
        self.minority = Counter(torch.flatten(self.train_tgt).tolist()).most_common()[-1][0]

        return self

    def infer(self, inp, use_torch=True):
        # inference with pytorch models
        if use_torch:
            self.model.eval()  # set model to eval before inference
            w = self.model(inp)
            # p = torch.sigmoid(w)
            # pred = proba > self.decision_boundary
        
        # inference with xgboost
        else:
            w = self.model.predict(inp.unsqueeze(0).numpy())

        return w

    def query(self, w, p):
        q_u = np.exp(-self.gamma * np.abs(w))  # uncertainty sampling
        q_m = 1 if self.model.classify(p) == self.minority else 0  # minority oversampling
        q = self.alpha * q_u + self.beta * q_m  # query probability

        return np.random.binomial(1, q)  # query decision via binary sampling

    def update(self):
        # get queried samples and indices to update for train and valid
        self.queried_inp, self.queried_tgt = map(lambda x: torch.cat(x, dim=0), zip(*self.queried))
        indices = torch.randperm(len(self.queried))
        valid_idx = indices[:int(len(indices) * self.val_ratio)]
        train_idx = indices[int(len(indices) * self.val_ratio):]

        # update train set
        self.train_inp = torch.cat([self.train_inp, self.queried_inp[train_idx]], dim=0)
        self.train_tgt = torch.cat([self.train_tgt, self.queried_tgt[train_idx]], dim=0)

        # update validation set
        self.valid_inp = torch.cat([self.valid_inp, self.queried_inp[valid_idx]], dim=0) 
        self.valid_tgt = torch.cat([self.valid_tgt, self.queried_tgt[valid_idx]], dim=0)

        # update minority label
        self.minority = Counter(torch.flatten(self.train_tgt).tolist()).most_common()[-1][0]

        # print info
        print(
            "===================================\n"
            f'Queried: {self._get_distribution(self.queried_tgt)}\n'
            f'Updated train: {self._get_distribution(self.train_tgt)}\n'
            f'Updated valid: {self._get_distribution(self.valid_tgt) if len(self.valid_tgt) > 0 else None}\n'
            f'Percentage: {len(self.train_tgt) / self.full_size:.2f}\n'
            "==================================="
        )

        # empty the list of queried samples
        self.queried.clear()

        # make train_loader and valid_loader
        self.train_loader = DataLoader(TensorDataset(self.train_inp, self.train_tgt), batch_size=len(self.train_tgt))
        self.val_loader = DataLoader(TensorDataset(self.valid_inp, self.valid_tgt), batch_size=len(self.valid_tgt)) if len(self.valid_tgt) > 0 else None
        return


class Experiment(object):
    
    def __init__(self, config: str) -> None:
        # get configurations
        self.config = get_config(config)
        
        # set seed
        pl.seed_everything(self.config.exp.seed)

        # prepare data
        self.dm = KEMOCONDataModule(
            config      = self.config.data,
            label_fn    = transform_label(self.config.exp.target, self.config.exp.pos_label),
        )

        # get experiment name
        self.exp_name = f'{self.config.exp.model}_{self.config.exp.type}_{self.config.exp.target}_{self.config.exp.pos_label}_{int(time.time())}'
        print(f'Experiment: {self.exp_name}, w/ participants: {list(self.dm.ids)}')

        # make directory to save results
        os.makedirs(os.path.expanduser(self.config.exp.savedir), exist_ok=True)

        # set path to save experiment results
        self.savepath = os.path.expanduser(os.path.join(self.config.exp.savedir, f'{self.exp_name}.json'))
        
    def init_logger(self, pid):
        # set version number if needed
        version = '' if pid is None else f'_{pid:02d}'

        # make logger
        logger = TensorBoardLogger(
            save_dir    = os.path.expanduser(self.config.logger.logdir),
            name        = f'{self.exp_name}',
            version     = version
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
        # init model
        self.model = self.init_model(self.config.hparams)

        # setup datamodule
        self.dm.setup(stage=None, test_id=pid)

        # init training with pl.LightningModule models
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

            # make trainer
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
            [results] = trainer.test(self.model)

            # return metrics and confusion matrix
            metr = {
                'pid': pid,
                'acc': results['test_acc'],
                'ap': results['test_ap'],
                'f1': results['test_f1'],
                'auroc': results['test_auroc'],
                'num_epochs': self.model.current_epoch,
            }
            cm = self.model.cm
        
        else:
            # train model: concat train and valid inputs and labels and convert torch tensors to numpy arrays
            X_train, y_train = map(lambda x: torch.cat(x, dim=0).numpy(), zip(self.dm.kemocon_train[:], self.dm.kemocon_val[:]))
            self.model.train(X_train, y_train)

            # test model
            X_test, y_test = map(lambda x: x.numpy(), self.dm.kemocon_test[:])
            metr, cm = self.model.test(X_test, y_test)

        return metr, cm

    def _active_body(self, pid=None):
        # init model 
        self.model = self.init_model(self.config.hparams)

        # init active learner and set it up for training
        self.learner = ActiveLearner(
            config      = self.config.exp.active_learning,
            model       = self.model,
            datamodule  = self.dm,
        ).setup(pid)
        
        # init dict to store results
        results = {'config': config_to_dict(self.config)}

        # init training with pl.LightningModule models
        if self.config.trainer is not None:
            # init logger
            if self.config.logger is not None:
                logger = self.init_logger(pid)

            # init lr monitor and checkpoint callback
            callbacks = list()
            if self.learner.val_ratio > 0:
                checkpoint_callback = ModelCheckpoint(
                    monitor     = 'valid_loss',
                    save_last   = True,
                    mode        = 'min',
                )
                callbacks.append(checkpoint_callback)

            # make trainer
            trainer_args = vars(self.config.trainer)
            trainer_args.update({
                'logger': logger,
                'callbacks': callbacks
            })
            trainer = pl.Trainer(**trainer_args)

            # fit model to initial batch
            trainer.fit(
                model               = self.model,
                train_dataloader    = self.learner.init_loader,
                val_dataloaders     = self.learner.val_loader
            )
            
            # test model and get results
            [metr] = trainer.test(
                model               = self.model,
                test_dataloaders    = self.dm.test_dataloader()
            )
            counts = dict(self.learner._get_counts())
            cm = self.model.cm
            print(trainer.checkpoint_callback.best_model_path)
            print(cm)
            
            # log test results for the initial batch
            results.setdefault('metrics', list()).append(metr)
            results.setdefault('counts', list()).append(counts)
            results.setdefault('cms', list()).append(cm.tolist())

            # now receive samples one by one from a stream
            for inp, tgt in self.learner.stream_loader:

                # infer label
                w = self.learner.infer(inp, use_torch=True).detach().cpu()
                p = torch.sigmoid(w)

                # query if condition is met
                if self.learner.query(w, p):
                    self.learner.queried.append((inp, tgt))  # add current sample to queried set

                # if the number of queried samples is larger than the update size
                if len(self.learner.queried) >= self.learner.update_size:
                    self.learner.update()  # update active learner

                    # reload model from the last best checkpoint if we are training with validation set
                    if self.learner.val_loader is not None:
                        self.model = self.model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

                    # update model to use a different learning rate for learning from datastream
                    if self.learner.update_lr is not None:
                        self.model.hparams.learning_rate = self.learner.update_lr

                    # re-fit model to the new trainset
                    trainer.max_epochs += self.learner.update_epochs
                    trainer.fit(
                        model               = self.model,
                        train_dataloader    = self.learner.train_loader,
                        val_dataloaders     = self.learner.val_loader
                    )
                    
                    # test fitted model again on test set
                    [metr] = trainer.test(
                        model               = self.model,
                        test_dataloaders    = self.dm.test_dataloader()
                    )
                    counts = dict(self.learner._get_counts())
                    cm = self.model.cm
                    print(trainer.checkpoint_callback.best_model_path)
                    print(cm)

                    # log test results
                    results['metrics'].append(metr)
                    results['counts'].append(counts)
                    results['cms'].append(cm.tolist())

            # TODO: now that we have seen all samples from the stream, do we want to do anything else?

        # active learning with XGBoost
        else:
            # get initial batch
            X_init, y_init = map(lambda x: torch.cat(x, dim=0).numpy(), zip(self.learner.init_batch[:], self.learner.val_batch[:]))

            # fit model to initial batch
            self.model.train(X_init, y_init)

            # test model and get results
            X_test, y_test = map(lambda x: x.numpy(), self.dm.kemocon_test[:])
            metr, cm = self.model.test(X_test, y_test)
            counts = dict(self.learner._get_counts())

            # save test results
            results.update({
                'metrics': [metr],
                'counts': [counts],
                'confmats': [cm.tolist()]
            })
            print(metr)
            print(cm)

            # get samples from a stream
            for inp, tgt in self.learner.datastream:

                # infer label
                w = self.learner.infer(inp, use_torch=False)
                p = 1 / (1 + np.exp(-w))
                
                # query if condition is met
                if self.learner.query(w, p):
                    self.learner.queried.append((inp.unsqueeze(0), tgt.unsqueeze(0)))

                # if queried the update size number of samples
                if len(self.learner.queried) >= self.learner.update_size:
                    # update train + val & minority label
                    # & reset queried samples buffer
                    self.learner.update()

                    # update model with queried samples
                    X_train, y_train = self.learner.train_inp.numpy(), self.learner.train_tgt.numpy()
                    self.model.train(X_train, y_train, model=self.model.bst)

                    # test updated model
                    metr, cm = self.model.test(X_test, y_test)
                    counts = dict(self.learner._get_counts())

                    # save results
                    results['metrics'].append(metr)
                    results['counts'].append(counts)
                    results['confmats'].append(cm.tolist())
                    print(metr)
                    print(cm)

        return results

    def run(self) -> None:
        # run holdout validation
        if self.config.exp.type == 'holdout':
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

        # run stream-based active learning (holdout)
        if self.config.exp.type == 'active-holdout':
            results = self._active_body()

        # save results
        with open(self.savepath, 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    # init parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to an experiment configuration file')
    args = parser.parse_args()

    # run experiment with configuration
    exp = Experiment(args.config)
    exp.run()

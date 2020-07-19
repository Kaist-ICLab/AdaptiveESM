import os
import time
import random
import numpy as np
from collections import namedtuple

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sacred import Experiment
from sacred.observers import MongoObserver

from dataset import DLBase, WrapperDataset
from model import LSTM
from tools import EarlyStopping


# configuration for sacred
EX_NAME = 'baseline'
DB_URL = 'localhost:27017'
DB_NAME = 'sacred'
ex = Experiment(EX_NAME)
ex.observers.append(MongoObserver.create(url=DB_URL, db_name=DB_NAME))


class Trainer(object):
    def __init__(self):
        self.paths = self.initialize()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.datasets = self.get_datasets()
        self.dataloaders = self.get_dataloaders()
        self.model = self.make_model()
        self.optimizer = self.make_optimizer()
        self.criterion = self.make_criterion()
        self.stopper = self.make_stopper()
        self.scheduler = self.make_scheduler()

    @ex.capture(prefix='training')
    def initialize(self, save_path, _seed):
        # set seeds for reproducibility
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        torch.cuda.manual_seed(_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # make folders to save model checkpoints
        os.makedirs(os.path.join(save_path, str(_seed)), exist_ok=True)
        paths = {
            'best': os.path.join(save_path, str(_seed), 'ckpt_best.pt'),
            'last': os.path.join(save_path, str(_seed), 'ckpt_last.pt')
        }
        return paths

    @ex.capture(prefix='dataset')
    def get_datasets(self, dataset_path, esm_path, target, uids):
        trainval = DLBase(dataset_path, esm_path, target, uids['trainval'])
        inputs, labels = list(zip(*[(inp, int(label > 0)) for inp, label in list(zip(trainval.inputs, trainval.labels)) if label != 0]))
        inputs, labels = np.array(inputs), np.array(labels)

        Fold = namedtuple('Fold', 'train, valid')
        Data = namedtuple('Data', 'inputs, labels')
        skf = StratifiedKFold(n_splits=2)
        folds = {i: Fold(
            Data(inputs[train_idx], labels[train_idx]),
            Data(inputs[valid_idx], labels[valid_idx])
            ) for i, (train_idx, valid_idx) in enumerate(skf.split(inputs, labels))}
        train_set = WrapperDataset(folds[0].train.inputs, folds[0].train.labels)
        valid_set = WrapperDataset(folds[0].valid.inputs, folds[0].valid.labels)

        test = DLBase(dataset_path, esm_path, target, uids['test'])
        inputs, labels = list(zip(*[(inp, int(label > 0)) for inp, label in list(zip(test.inputs, test.labels)) if label != 0]))
        inputs, labels = np.array(inputs), np.array(labels)
        test_set = WrapperDataset(inputs, labels)
        return {'train': train_set, 'val': valid_set, 'test': test_set}

    @ex.capture(prefix='training')
    def get_dataloaders(self, batch_size):
        train_loader = DataLoader(dataset=self.datasets['train'],
                                  batch_size=batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(dataset=self.datasets['val'],
                                  batch_size=batch_size,
                                  shuffle=False)
        test_loader = DataLoader(dataset=self.datasets['test'],
                                 batch_size=batch_size,
                                 shuffle=False)

        print('Train: ', np.unique(train_loader.dataset.labels, return_counts=True))
        print('Valid: ', np.unique(valid_loader.dataset.labels, return_counts=True))
        print('Test: ', np.unique(test_loader.dataset.labels, return_counts=True))
        return {'train': train_loader, 'val': valid_loader, 'test': test_loader}

    @ex.capture(prefix='training')
    def make_model(self, input_size, output_size, hidden_size, num_layers, p_drop):
        model = LSTM(input_size, output_size, hidden_size, num_layers, p_drop).to(self.device)
        return model

    @ex.capture(prefix='training')
    def make_optimizer(self, learning_rate):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer

    @ex.capture(prefix='training')
    def make_criterion(self, prediction_mode):
        if prediction_mode == 'regression':
            criterion = nn.SmoothL1Loss()
        elif prediction_mode == 'classification':
            # criterion = nn.CrossEntropyLoss()
            criterion = nn.BCEWithLogitsLoss()
        return criterion

    @ex.capture(prefix='stopper')
    def make_stopper(self, patience, verbose, delta):
        stopper = EarlyStopping(patience, verbose, delta, self.paths['best'])
        return stopper

    @ex.capture(prefix='scheduler')
    def make_scheduler(self, T_0, T_mult):
        scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0, T_mult)
        return scheduler

    @ex.capture(prefix='training')
    def get_preds(self, outputs, prediction_mode):
        if prediction_mode == 'regression':
            preds = torch.round(outputs)
        elif prediction_mode == 'classification':
            # _, preds = torch.max(outputs, 1)
            preds = torch.round(torch.sigmoid(outputs))
        return preds

    @ex.capture(prefix='training')
    def train(self, epoch, _run):
        self.model.train()
        running_losses, running_corrects = 0.0, 0
        for i, (inputs, labels) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()
            loss = self.criterion(outputs, labels.type_as(outputs))
            _run.log_scalar('train loss', float(loss.data))

            # backward and optimize
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(epoch+i / len(self.dataloaders['train']))
            # clip_grad_norm_(model.parameters(), hp.learning_rate)
            preds = self.get_preds(outputs)

            running_losses += loss.item()
            running_corrects += torch.sum(preds == labels).item()
            # print(f'train loss = {loss.item():.6f}', np.unique(preds.cpu().detach(), return_counts=True))
        loss = running_losses / len(self.dataloaders['train'])
        accuracy = running_corrects / len(self.dataloaders['train'].dataset)
        return loss, accuracy

    @ex.capture(prefix='training')
    def validate(self, output_size, _run):
        self.model.eval()
        with torch.no_grad():
            running_losses, running_corrects = 0.0, 0
            confmat = np.zeros((output_size+1, output_size+1))
            for inputs, labels in self.dataloaders['val']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels.type_as(outputs))
                _run.log_scalar('valid loss', float(loss.data))
                preds = self.get_preds(outputs)

                running_losses += loss.item()
                running_corrects += torch.sum(preds == labels).item()
                confmat += confusion_matrix(labels.cpu(), preds.cpu())
                # print(f'val loss = {loss.item():.6f}', np.unique(preds.cpu().detach(), return_counts=True))
        loss = running_losses / len(self.dataloaders['val'])
        accuracy = running_corrects / len(self.dataloaders['val'].dataset)
        return loss, accuracy, confmat

    @ex.capture(prefix='training')
    def test(self, output_size):
        self.model.load_state_dict(torch.load(self.paths['best']))
        self.model.eval()
        with torch.no_grad():
            running_losses, running_corrects = 0.0, 0
            confmat = np.zeros((output_size+1, output_size+1))
            for inputs, labels in self.dataloaders['test']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels.type_as(outputs))
                preds = self.get_preds(outputs)

                running_losses += loss.item()
                running_corrects += torch.sum(preds == labels).item()
                confmat += confusion_matrix(labels.cpu(), preds.cpu())
        loss = running_losses / len(self.dataloaders['test'])
        accuracy = running_corrects / len(self.dataloaders['test'].dataset)
        return loss, accuracy, confmat

    @ex.capture(prefix='training')
    def run(self, num_epochs, print_every, multi_gpu, _run):
        start = time.time()
        for epoch in range(0, num_epochs):
            tic = time.time()
            if (epoch+1) % print_every == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]', end=':\t')

            train_loss, train_acc = self.train(epoch=epoch)
            val_loss, val_acc, val_confmat = self.validate()

            toc = time.time()
            if (epoch+1) % print_every == 0:
                print(f'{toc - tic:.2f}s/epoch | Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val. loss: {val_loss:.4f}, acc: {val_acc:.4f}')
                print(f'Confusion matrix:\n{val_confmat}')

            self.stopper(val_loss, self.model)
            if self.stopper.early_stop:
                print('Early stopping')
                break

        end = time.time()
        if multi_gpu:
            torch.save(self.model.module.state_dict(), self.paths['last'])
        else:
            torch.save(self.model.state_dict(), self.paths['last'])
        
        print(f'Training finished in {end - start:.2f}s for {epoch+1} epochs...')
        test_loss, test_acc, test_confmat = self.test()
        return test_loss, test_acc, test_confmat


@ex.config
def get_config():
    training = {
        'batch_size': 500,
        'input_size': 7,
        'output_size': 1,
        'hidden_size': 500,
        'num_layers': 5,
        'p_drop': 0.5,
        'learning_rate': 0.01,
        'prediction_mode': 'classification',
        'num_epochs': 500,
        'print_every': 1,
        'multi_gpu': False,
        'save_path': '/home/coder/projects/dailyLife2/codes/models'
    }
    # dataset_path, esm_path, target, uids
    dataset = {
        'dataset_path': '/home/coder/data/dailyLife2/datasets/baseline',
        'esm_path': '/home/coder/data/dailyLife2/metadata/esm_data.csv',
        'uids': {
            'trainval': [3024, 3029, 3025, 3028, 3027],
            'test': [3012],
        },
        'target': 'arousal',
    }
    stopper = {
        'patience': 100, 
        'verbose': True,
        'delta': 0,
    }
    scheduler = {
        'T_0': 10,
        'T_mult': 2,
    }


@ex.main
def main(_run):
    trainer = Trainer()
    loss, accuracy, confmat = trainer.run()

    return {'loss': loss, 'accuracy': accuracy, 'confusion matrix': confmat}


if __name__ == "__main__":
    ex.run_commandline()

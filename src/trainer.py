from abc import abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils import clip_grad_norm_

from conn.conn import CONN
from conn.crystal_conn import CrystalCONN
from early_stopper import EarlyStopper

class Trainer():
    
    def __init__(
                self,
                model: CONN,
                dataset: Dataset,
                optimizer: optim.Optimizer = optim.SGD,
                epochs = 1000,
                batch_size = 64,
                train_params: dict[str,float] = {'lr' : 1.0e-3},
                device = torch.device('cpu')
                ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.__model = model
        self.__dataset = dataset
        self.__optimizer: optim.Optimizer = optimizer(model.parameters(),**train_params)

        self.__train_loader, self.__val_loader = model.configure_dataloaders(dataset)
        self.early_stopper = EarlyStopper()

        self.grad_clip_threshold = 5

    def fit(self):
        for epoch in range(self.epochs):
            train_loss, val_loss = self.fit_epoch()
            
            print(f'Training loss: {train_loss}')
            print(f'Validation loss: {val_loss}')

            if self.early_stopper.should_stop(val_loss):
                print(f'Reached early stopping criteria!')
                print(f'Minimum validation loss: {self.early_stopper.min_loss}')
                return

    def fit_epoch(self):
        train_loss = 0
        val_loss = 0

        self.__model.train()
        for in_batch,target_batch,len_batch in self.__train_loader:

            self.__optimizer.zero_grad()
            h = None
            stress = torch.zeros((in_batch.shape[0],9),device=self.device)
            preds = []
            for t in range(in_batch.shape[1]):
                pred, h = self.__model(in_batch[:,t,:],h)
                stress = in_batch[:,t,CrystalCONN.__INS['stress']] + pred[:,CrystalCONN.__OUTS['dstress']]
                if t < in_batch.shape[1]-1:
                    in_batch[:,t+1,CrystalCONN.__INS['stress']] = stress
                
                preds.append(pred)
            
            preds = torch.stack(preds)
            preds = torch.transpose(preds,dim0=1,dim1=0)
            
            loss = self.__model.loss(in_batch,preds)

            clip_grad_norm_(self.__model.parameters(), max_norm=self.grad_clip_threshold)

            loss.backward()

            self.__optimizer.step()

        self.__model.eval()
        with torch.no_grad():
            for in_batch,target_batch,len_batch in self.__val_loader:

                h = None
                stress = torch.zeros((in_batch.shape[0],9),device=self.device)
                preds = []
                for t in range(in_batch.shape[1]):
                    pred, h = self.__model(in_batch[:,t,:],h)
                    stress = in_batch[:,t,CrystalCONN.__INS['stress']] + pred[:,CrystalCONN.__OUTS['dstress']]
                    if t < in_batch.shape[1]-1:
                        in_batch[:,t+1,CrystalCONN.__INS['stress']] = stress

                    preds.append(pred)

                preds = torch.stack(preds)
                preds = torch.transpose(preds,dim0=1,dim1=0)

                val_loss += self.__model.loss(in_batch,preds)

        return train_loss, val_loss
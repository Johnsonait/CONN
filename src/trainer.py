from abc import abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np

from .conn.conn import CONN
from .conn.crystal_conn import CrystalCONN
from .early_stopper import EarlyStopper

class Trainer():
    
    def __init__(
                self,
                model: CONN,
                train_dataset: Dataset,
                val_dataset: Dataset,
                optimizer: optim.Optimizer = optim.SGD,
                epochs = 1000,
                batch_size = 64,
                train_params = {'lr' : 1.0e-3},
                device = torch.device('cpu')
                ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.__model = model
        self.__train_dataset = train_dataset
        self.__val_dataset = val_dataset
        self.__optimizer: optim.Optimizer = optimizer(model.parameters(),**train_params)

        self.__train_loader, self.__val_loader = model.configure_dataloaders(train_dataset,val_dataset,batch_size)
        self.early_stopper = EarlyStopper()

        self.grad_clip_threshold = 5
        self.training_history = []
        self.validation_history = []

    def __save_histories(self):
        np.savetxt('training/training.txt',np.array(self.training_history))
        np.savetxt('training/validation.txt',np.array(self.validation_history))

        pass

    def fit(self):
        print('\t\tNet Loss \tCoAM Loss \tEnergy Loss \tConstitutive Loss \tTarget Loss')
        for epoch in range(self.epochs):
            self.fit_epoch()

            self.__model.save('./model.pt')

            self.__save_histories()
            print(f'Training  : {self.training_history[-1]}')
            print(f'Validation: {self.validation_history[-1]}')

            if self.early_stopper.should_stop(self.validation_history[-1][0]):
                print(f'Reached early stopping criteria!')
                print(f'Minimum validation loss: {self.early_stopper.min_loss}')
                return

    def fit_epoch(self):
        train_loss = E_Loss_t = CoAM_Loss_t = CONN_Loss_t = target_Loss_t = 0
        val_loss = E_Loss_v = CoAM_Loss_v = CONN_Loss_v = target_Loss_v = 0
        self.__model.train()
        batch_count = 0
        for in_batch,target_batch,len_batch in self.__train_loader:
            batch_count += 1

            self.__optimizer.zero_grad()
            h = None
            stress = None
            preds = []
            stresses = []
            for t in range(in_batch.shape[1]):
                out,stress, h = self.__model(in_batch[:,t,:],stress,h)

                stresses.append(stress)
                preds.append(out)
            
            preds = torch.stack(preds)
            stresses = torch.stack(stresses)
            preds = torch.transpose(preds,dim0=1,dim1=0)
            stresses = torch.transpose(stresses ,dim0=1,dim1=0)
            
            loss,CoAM,E,CONN,target = self.__model.fast_loss(in_batch,preds,stresses,target_batch)
            train_loss    += loss.item()
            CoAM_Loss_t   += CoAM.item()
            E_Loss_t      += E.item()
            CONN_Loss_t   += CONN.item()
            target_Loss_t += target.item()
            
            loss.backward()

            clip_grad_norm_(self.__model.parameters(), max_norm=self.grad_clip_threshold)

            self.__optimizer.step()
        
        train_loss /= batch_count
        CoAM_Loss_t   /= batch_count
        E_Loss_t      /= batch_count
        CONN_Loss_t   /= batch_count
        target_Loss_t /= batch_count
        self.training_history.append([train_loss,CoAM_Loss_t,E_Loss_t,CONN_Loss_t,target_Loss_t])

        self.__model.eval()
        batch_count = 0
        with torch.no_grad():
            for in_batch,target_batch,len_batch in self.__val_loader:
                batch_count += 1

                h = None
                stress = None
                preds = []
                stresses = []
                for t in range(in_batch.shape[1]):
                    out,stress, h = self.__model(in_batch[:,t,:],stress,h)

                    stresses.append(stress)
                    preds.append(out)

                preds = torch.stack(preds)
                stresses = torch.stack(stresses)
                preds = torch.transpose(preds,dim0=1,dim1=0)
                stresses = torch.transpose(stresses ,dim0=1,dim1=0)

                loss,CoAM,E,CONN,target = self.__model.fast_loss(in_batch,preds,stresses,target_batch)
                val_loss      += loss.item()
                CoAM_Loss_v   += CoAM.item()
                E_Loss_v      += E.item()
                CONN_Loss_v   += CONN.item()
                target_Loss_v += target.item()

        val_loss /= batch_count
        CoAM_Loss_v   /= batch_count
        E_Loss_v      /= batch_count
        CONN_Loss_v   /= batch_count
        target_Loss_v /= batch_count
        self.validation_history.append([val_loss,CoAM_Loss_v,E_Loss_v,CONN_Loss_v,target_Loss_v])
        
        return 
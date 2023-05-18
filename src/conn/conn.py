from abc import abstractmethod


import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset


class CONN(nn.Module):

    def __init__(self):
        super().__init__()
        pass
    
    @abstractmethod
    def forward(self,inputs):
        raise NotImplementedError

    @abstractmethod
    def loss(self,inputs,outputs):
        raise NotImplementedError
    
    @abstractmethod
    def fast_loss(self,inputs,outputs):
        raise NotImplementedError

    @abstractmethod
    def constitutive_eqn(self,**vars):
        raise NotImplementedError
    
    @abstractmethod 
    def energy_eqn(self,**vars):
        raise NotImplementedError

    @abstractmethod
    def save(self,dir):
        raise NotImplementedError

    @abstractmethod
    def load(self,dir):
        raise NotImplementedError
    
    @abstractmethod
    def configure_dataloaders(self):
        raise NotImplementedError
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

from conn.conn import CONN

class Trainer():
    
    def __init__(
                self,
                model: CONN,
                dataset: Dataset,
                optimizer: optim.Optimizer = optim.SGD,
                params: dict[str,float] = {'lr' : 1.0e-3}
                ):
        self.__model = model
        self.__dataset = dataset
        self.__optimizer = optimizer(model.parameters(),**params)

    def fit(self):
        pass

    def fit_epoch(self):
        pass
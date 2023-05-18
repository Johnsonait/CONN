import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from conn.conn import CONN
from conn.conn_dataset import CONNDataset

class ModelTester():

    def __init__(self,model: CONN,dataset: CONNDataset):
        self.__model = model
        self.__dataset = dataset

    def run_model(self):
        
        

        return
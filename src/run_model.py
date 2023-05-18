import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .conn.conn import CONN
from .conn.conn_dataset import CONNDataset

class ModelTester():

    def __init__(self,model: CONN,dataset: CONNDataset):
        self.__model = model
        self.__dataset = dataset
        self.__input_scaler = dataset.input_scaler
        self.__target_scaler = dataset.target_scaler

    def run_model(self):
        inputs,targets,lens = self.__dataset[0]

        h = None
        stress = None
        preds = []
        stresses = []
        self.__model.eval()
        with torch.no_grad():
            for t in range(inputs.shape[0]):
                out,stress, h = self.__model(inputs[t,:],stress,h)

                stresses.append(stress)
                preds.append(out)

            preds = torch.stack(preds)
            stresses = torch.stack(stresses)
            preds = torch.transpose(preds,dim0=1,dim1=0).squeeze(0)
            stresses = torch.transpose(stresses ,dim0=1,dim1=0).squeeze(0)

            stresses = stresses[:,:9]#self.__scaler_inv(stresses[:,:6],slice(0,6))
            targ_stresses = targets[:,:9]#self.__scaler_inv(targets[:,:6],slice(0,6))

            plt.plot(stresses[:,:],label='Model Output')
            plt.plot(targ_stresses[:,:],label='Target',color='black',linestyle='--')
            plt.show()

        return


    def __scaler_inv(self,values, index):
        mins = self.__target_scaler.data_min_
        maxes = self.__target_scaler.data_max_

        min_val = torch.tensor(mins[index])
        max_val = torch.tensor(maxes[index])

        values[:] = 2*(values[:]+1)*(max_val-min_val) + min_val

        return values
import torch
from torch.utils.data import Dataset
import h5py

class CONNDataset(Dataset):

    train_group = 'training'
    val_group = 'validation'
    test_group = 'test'

    input_label = 'inputs'
    target_label = 'targets'
    len_label = 'seq_lengths'

    def __init__(
                 self,
                 dir = '../../data',
                 file_name = 'dataset.hdf5',
                 device = torch.device('cpu')
                 ):
        self.__dir = dir
        self.__file_name = file_name
        self.device = device

        self.__dataset = h5py.File(f'{dir}/{file_name}','r')

        self.__training_set = self.__dataset[CONNDataset.train_group]
        self.__validation_set = self.__dataset[CONNDataset.val_group]

        


    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, index):
        input = self.inputs[index][:]
        output = self.targets[index][:]
        lengths = self.lens[index]
        return input,output,lengths

import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
                 dir = './data',
                 file_name = 'dataset.hdf5',
                 split = 'training',
                 scalers = None,
                 device = torch.device('cpu')
                 ):
        self.__dir = dir
        self.__file_name = file_name
        self.device = device
        feature_range = (-1,1)
        
        must_fit = False
        if scalers is not None:
            self.input_scaler=scalers['input']
            self.target_scaler=scalers['target']
        else:
            must_fit = True
            self.input_scaler =  MinMaxScaler(feature_range=feature_range)
            self.target_scaler = MinMaxScaler(feature_range=feature_range)

        self.__dataset = h5py.File(f'{dir}/{file_name}','r')

        self.__data = Dataset(self.__dataset[split],device=device)
        self.inputs = self.__data[CONNDataset.input_label]
        self.targets = self.__data[CONNDataset.target_label]
        self.lens = self.__data[CONNDataset.len_label]

        # Assuming X is a three-dimensional array with shape (num_sequences, seq_length, num_features)
        in_sequences, in_seq_length, in_features = self.inputs.shape
        target_sequences, target_seq_length, target_features = self.targets.shape
        # Reshape the three-dimensional array into a two-dimensional array
        in_2d = self.inputs.reshape((in_sequences * in_seq_length, in_features))
        target_2d = self.targets.reshape((target_sequences * target_seq_length, target_features))

        in_mean = np.mean(in_2d)
        in_std = np.std(in_2d)
        target_mean = np.mean(target_2d)
        target_std = np.std(target_2d)
        # Find sequences with outlier values
        outlier_indices = []
        # Number of std-devs away from mean before somthing is considered an 
        # outlier
        dev_num = 5
        for i in range(in_sequences):
            seq = self.inputs[i, :, :]
            targ_seq = self.targets[i, :, :]
            seq_outliers = np.where(np.abs(seq - in_mean) > dev_num * in_std)
            targ_outliers = np.where(np.abs(targ_seq - target_mean) > dev_num * target_std)
            if len(seq_outliers[0]) > 0 or len(targ_outliers[0]) > 0:
                outlier_indices.append(i)
        
        # Delete outlier sequences
        self.inputs = np.delete(self.inputs, outlier_indices, axis=0)
        self.targets = np.delete(self.targets, outlier_indices, axis=0)
        self.lens = np.delete(self.lens, outlier_indices, axis=0)

        # Reshape data back to 3D arrays
        self.inputs = self.inputs.reshape((self.inputs.shape[0], in_seq_length, in_features))
        self.targets = self.targets.reshape((self.targets.shape[0], target_seq_length, target_features))
        #Reshape again for normalization
        in_sequences, in_seq_length, in_features = self.inputs.shape
        target_sequences, target_seq_length, target_features = self.targets.shape
        in_2d = self.inputs.reshape((in_sequences * in_seq_length, in_features))
        target_2d = self.targets.reshape((target_sequences * target_seq_length, target_features))

        self.sample_count = self.lens.shape[0]
        print(f'{self.split} samples: {self.sample_count}')
        
        # Fit the scaler to the two-dimensional array of features
        if must_fit:
            #self.input_scaler.fit(in_2d[:,8:])
            self.input_scaler.fit(in_2d)
            self.target_scaler.fit(target_2d)
        # Scale and reshape the scaled two-dimensional array back into the original three-dimensional shape
        # NOTE: if DT is const, remove from transform (featured index 0) to avoid nan values
        in_scaled = in_2d
        #in_scaled[:,8:] = self.input_scaler.transform(in_2d[:,8:])
        in_scaled = self.input_scaler.transform(in_2d)
        target_scaled = self.target_scaler.transform(target_2d)
        self.inputs = in_scaled.reshape((in_sequences, in_seq_length, in_features))
        self.targets = target_scaled.reshape((target_sequences, target_seq_length, target_features))

        self.inputs = torch.tensor(self.inputs).to(device=self.device)
        self.targets = torch.tensor(self.targets).to(device=self.device)
        self.lens = torch.tensor(self.lens).to(device=self.device)

        if split == 'training':
            pickle.dump(self.input_scaler,open('./data/attributes/input_scaler.sav','wb'))
            pickle.dump(self.target_scaler,open('./data/attributes/target_scaler.sav','wb'))

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, index):
        input = self.inputs[index][:]
        output = self.targets[index][:]
        lengths = self.lens[index]
        return input,output,lengths

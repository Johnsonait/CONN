import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence

from conn.conn import CONN
from conn.crystal import Crystal
from conn_dataset import CONNDataset

class CrystalCONN(CONN):

    __n_slip = Crystal.n_slip

    __INS: dict[str, slice] = {
        'dt' : slice(0,1),
        'L' : slice(1,10),
        'stress': slice(10,19)
    }

    __in_sz = 1+9+9

    __OUTS: dict[str,slice] = {
        'dstress' : slice(0,9),
        'dslip' : slice(9,9+__n_slip),
        'tau' : slice(9+__n_slip, 9+2*__n_slip),
        'Le' : slice(9+2*__n_slip,9+2*__n_slip+9)
    }

    __out_sz = 9 + 2*__n_slip + 9

    def __init__(
                 self,
                 hidden_size=128,
                 loss_fcn = nn.MSELoss,
                 device=torch.device('cpu')):
        super().__init__(self)
        self.hidden_size = hidden_size
        self.device = device
        self.loss_fcn = loss_fcn
        self.__crystal: Crystal = Crystal()

        self.GRU = nn.GRU(
            input_size = CrystalCONN.__in_sz,
            hidden_size = hidden_size,
            num_layers = 1,
            device = device
        )

        self.output_layer = nn.Linear(hidden_size,CrystalCONN.__out_sz,device=device)
    
    def forward(self, input,h=None):

        out, h = self.GRU(input,h)

        out = self.output_layer(h)

        return out,h

    
    def loss(self, inputs, outputs):
        
        #inputs = packed_inputs.data
        #outputs = packed_outputs.data

        loss = 0
        # Batch loop
        batch_count = 0
        for k in range(inputs.shape[0]):
            batch_count += 1

            # Time loop
            time_count = 0
            for n in range(inputs.shape[1]):
                time_count += 1

                # Slice out the input tensors
                dt = inputs[k,n,CrystalCONN.__INS['dt']]
                L = inputs[k,n,CrystalCONN.__INS['L']]
                stress_in = inputs[k,n,CrystalCONN.__INS['stress']]

                # Slice out the outputs tensors
                stress_out = outputs[k,n,CrystalCONN.__OUTS['stress']]
                dstress = outputs[k,n,CrystalCONN.__OUTS['dstress']]
                dslip = outputs[k,n,CrystalCONN.__OUTS['dslip']]
                tau = outputs[k,n,CrystalCONN.__OUTS['tau']]
                Le = outputs[k,n,CrystalCONN.__OUTS['Le']]

                De = self.__sym(Le)

                loss += torch.abs(torch.sum(self.symmetry_eqn(stress=stress_out)))
                loss += torch.abs(
                                  self.energy_eqn(
                                        dt=dt,
                                        stress=stress_in,
                                        L=L,
                                        De=De,
                                        tau=tau,
                                        dslip=dslip)
                                  )
                loss += torch.abs(
                                  torch.sum(
                                            self.constitutive_eqn(
                                                dt=dt,
                                                stress=stress_in,
                                                dstress=dstress,
                                                L=L,
                                                dslip=dslip)
                                            )
                                  )
            # End of time loop

            loss /= time_count

        # End of batch loop
        loss /= batch_count

        return loss
    
    def symmetry_eqn(self,stress):
        return stress-self.__transpose(stress)
    
    def constitutive_eqn(self,dt,stress,dstress,L,dslip):

        D = self.__sym(L)
        omega = self.__asym(L)

        ret: torch.Tensor = dstress
        ret -= self.__contract(omega*dt,stress)
        ret += self.__contract(stress,omega*dt)
        ret -= self.__ELAS_ddot_A(A=D*dt,crystal=self.__crystal)
        for sys in range(self.__n_slip):
            PA = torch.tensor(self.__crystal.PA[sys,:],device=self.device).view(9)
            WA = torch.tensor(self.__crystal.WA[sys,:],device=self.device).view(9)
            ret += self.__ELAS_ddot_A(A=PA,crystal=self.crystal)
            ret += self.__contract(WA,stress)
            ret -= self.__contract(stress,WA)
            ret *= dslip
        ret += stress*self.__trace(D*dt)

        return ret
    
    def energy_eqn(self,dt,stress,L,De,tau,dslip):

        D = self.__sym(L)
        
        ret = torch.dot(stress,D*dt)
        ret -= torch.dot(stress,De*dt)
        ret -= torch.dot(tau,dslip)

        return ret

    def configure_dataloaders(self, train_set: CONNDataset, val_set: CONNDataset,batch_size):
        train_loader = DataLoader(train_set,batch_size=batch_size)
        val_loader = DataLoader(val_set,batch_size=batch_size)
        
        return train_loader, val_loader

    def __sym(self,A: torch.Tensor):
        return 0.5*(A + self.__transpose(A))

    def __asym(self,A):
        return 0.5*(A - self.__transpose(A))
        
    def __contract(self,A: torch.Tensor,B: torch.Tensor):
        return torch.matmul(A.view(3,3),B.view(3,3)).view(9)

    def __transpose(self,A: torch.Tensor):
        return A.view(3,3).transpose(0,1).view(9)

    def __trace(self,A: torch.Tensor):
        return A.view(3,3)[::4].sum()
    
    def __ELAS_ddot_A(A: torch.Tensor,crystal: Crystal):
        ret = A
        ret[:-3] = torch.matmul(crystal.ELAS,ret[:-3])
        ret[-3:] = ret[-6:-3]
        return ret
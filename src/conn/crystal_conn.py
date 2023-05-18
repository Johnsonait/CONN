import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence

from ..conn.conn import CONN
from ..conn.crystal import Crystal
from ..conn.conn_dataset import CONNDataset

class CrystalCONN(CONN):

    __n_slip = Crystal.n_slip

    __INS  = {
        'dt' : slice(0,1),
        'L' : slice(1,10),
        'stress' : slice(10,19)
    }

    __in_sz = 1+9

    __OUTS = {
        'stress' : slice(0,9),
        'dstress' : slice(9,18),
        'dslip' : slice(18,18+__n_slip),
        'tau' : slice(18+__n_slip, 18+2*__n_slip),
        'Le' : slice(18+2*__n_slip,18+2*__n_slip+9)
    }

    __out_sz = 9 + 9 + 2*__n_slip + 9

    __STRESS= slice(0,9)

    __stress_sz = 9

    def __init__(
                 self,
                 hidden_size=128,
                 loss_fcn = nn.MSELoss,
                 device=torch.device('cpu')):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.loss_fcn = loss_fcn()
        self.__crystal: Crystal = Crystal()

        self.GRU = nn.GRUCell(
            input_size = CrystalCONN.__in_sz+CrystalCONN.__stress_sz,
            hidden_size = hidden_size,
            device = device
        )

        self.output_layer = nn.Linear(hidden_size,CrystalCONN.__out_sz,device=device)

        self.double()

    def __init_hidden(self,batch_size):
        return torch.zeros((batch_size,self.hidden_size),device=self.device,dtype=torch.float64,requires_grad=True)

    def __init_stress(self,batch_size):
        return torch.zeros((batch_size,self.__stress_sz),device=self.device,dtype=torch.float64,requires_grad=True)
    
    def forward(self, x: Tensor,stress: Tensor=None,h: Tensor=None):
        assert x.dim() in (1,2), \
            f"CONN: Expected input to be 1-D or 2-D but received {x.dim()}-D tensor"
        is_batched = x.dim() == 2

        if not is_batched:
            x = x.unsqueeze(0)
        batch_size = x.size(0)

        if stress is None:
            stress = self.__init_stress(batch_size)

        h_out = self.GRU(torch.cat((x[:,:-9],stress),dim=1),h)

        out = self.output_layer(h_out)

        stress = (out[:,CrystalCONN.__OUTS['dstress']]+x[:,CrystalCONN.__INS['stress']]).clone()

        return out,stress,h_out
    
    def save(self, dir):
        torch.save(self.state_dict(), dir)
        return

    def load(self,params,dir):
        model = CrystalCONN(**params)
        model.load_state_dict(torch.load(dir,map_location=torch.device('cpu')))
        return model

    def fast_loss(self, inputs, outputs, stresses, targets):
        """This is a vectorized form of the loss function to speed up computation
        on GPUs"""
        dt = inputs[:, :, CrystalCONN.__INS['dt']]
        L = inputs[:, :, CrystalCONN.__INS['L']]
        stress_in = inputs[:, :, CrystalCONN.__INS['stress']]

        dstress = outputs[:, :, CrystalCONN.__OUTS['dstress']]
        dslip = outputs[:, :, CrystalCONN.__OUTS['dslip']]
        tau = outputs[:, :, CrystalCONN.__OUTS['tau']]
        Le = outputs[:, :, CrystalCONN.__OUTS['Le']]

        stress_out = stresses[:, :, CrystalCONN.__STRESS]
        target_stress = targets[:, :, CrystalCONN.__OUTS['stress']]

        De = self.__sym(Le)

        target_Loss = self.loss_fcn(stress_out, target_stress)

        CoAM_Loss = torch.square(torch.mean(self.symmetry_eqn(stress=stress_out), dim=(1, 2))).mean()
        E_Loss = torch.square(self.energy_eqn(dt, stress_in, L, De, tau, dslip)).mean()
        CONN_Loss = torch.square(torch.mean(self.constitutive_eqn(dt, stress_in, dstress, L, dslip), dim=(1, 2))).mean()

        loss = target_Loss + CoAM_Loss + E_Loss + CONN_Loss
        loss = loss

        return loss, CoAM_Loss, E_Loss, CONN_Loss, target_Loss

    def loss(self, inputs, outputs, stresses,targets):
        """This is a VERY slow way to evaulate the loss. It is here as a first
        step in understanding how the training is done since it makes the 
        intended evaluation of the loss explicit without introducing some
        potentially confusing vectorization measures. fast_loss() performs
        this vectorization and is MUCH faster because of it but perhaps slightly
        less readable."""
        
        loss = E_Loss = CoAM_Loss = CONN_Loss = target_Loss = 0
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
                dstress = outputs[k,n,CrystalCONN.__OUTS['dstress']]
                dslip = outputs[k,n,CrystalCONN.__OUTS['dslip']]
                tau = outputs[k,n,CrystalCONN.__OUTS['tau']]
                Le = outputs[k,n,CrystalCONN.__OUTS['Le']]

                stress_out = stresses[k,n,CrystalCONN.__STRESS]
                target_stress = targets[k,n,CrystalCONN.__OUTS['stress']]

                De = self.__sym(Le)

                target_Loss = self.loss_fcn(stress_out,target_stress)
                loss = loss + target_Loss
                target_Loss = target_Loss.item()

                CoAM_Loss = torch.abs(torch.sum(self.symmetry_eqn(stress=stress_out)))
                loss = loss + CoAM_Loss
                CoAM_Loss = CoAM_Loss.item()

                E_Loss =  torch.abs(
                                self.energy_eqn(
                                      dt=dt,
                                      stress=stress_in,
                                      L=L,
                                      De=De,
                                      tau=tau,
                                      dslip=dslip)
                                )
                loss = loss + E_Loss
                E_Loss = E_Loss.item()

                CONN_Loss = torch.sum(
                                torch.abs(
                                          self.constitutive_eqn(
                                              dt=dt,
                                              stress=stress_in,
                                              dstress=dstress,
                                              L=L,
                                              dslip=dslip)
                                          )
                                )
                loss = loss + CONN_Loss
                CONN_Loss = CONN_Loss.item()
            # End of time loop

            loss /= time_count

        # End of batch loop
        loss /= batch_count

        return loss,CoAM_Loss,E_Loss,CONN_Loss,target_Loss
    
    def symmetry_eqn(self,stress):
        stress = stress-self.__transpose(stress)
        return stress
    
    def constitutive_eqn(self,dt,stress,dstress,L,dslip):

        dD = self.__sym(L)*dt
        domega = self.__asym(L)*dt

        ret: torch.Tensor = dstress
        cont = self.__contract(domega,stress).clone()
        ret = ret - cont
        ret = ret + self.__contract(stress,domega).clone()
        ret = ret - self.__ELAS_ddot_A(A=dD,crystal=self.__crystal).clone()

        PA_expanded = torch.zeros((ret.shape[0],ret.shape[1],9),device=self.device,dtype=torch.float64)
        WA_expanded = torch.zeros((ret.shape[0],ret.shape[1],9),device=self.device,dtype=torch.float64)
        for sys in range(self.__n_slip):
            
            PA = torch.tensor(self.__crystal.PA[sys,:],device=self.device,dtype=torch.float64).view(9)
            WA = torch.tensor(self.__crystal.WA[sys,:],device=self.device,dtype=torch.float64).view(9)

            PA_expanded[:,:] = PA
            WA_expanded[:,:] = WA
            ret = ret + self.__ELAS_ddot_A(A=PA_expanded,crystal=self.__crystal).clone()
            ret = ret + self.__contract(WA_expanded,stress).clone()
            ret = ret - self.__contract(stress,WA_expanded).clone()
            ret = torch.mul(ret[:,:],dslip[:,:,sys].unsqueeze(-1)) 
        ret = ret + torch.mul(stress[:,:],self.__trace(dD).unsqueeze(-1))

        return ret
    
    def energy_eqn(self, dt, stress, L, De, tau, dslip):
        D = self.__sym(L)
        dD = torch.mul(D[:,:],dt[:,:])
        dDe = torch.mul(De[:,:],dt[:,:])

        ret = torch.einsum('...k,...k->...',stress,dD)
        ret = ret - torch.einsum('...k,...k->...',stress,dDe)
        ret = ret - torch.einsum('...k,...k->...',tau,dslip)

        return ret

    def configure_dataloaders(self, train_set: CONNDataset, val_set: CONNDataset,batch_size):
        train_loader = DataLoader(train_set,batch_size=batch_size)
        val_loader = DataLoader(val_set,batch_size=batch_size)
        
        return train_loader, val_loader

    def __sym(self,A: torch.Tensor):
        return 0.5*(A + self.__transpose(A)).clone()

    def __asym(self,A):
        return 0.5*(A - self.__transpose(A)).clone()

    def __contract(self, A: torch.Tensor, B: torch.Tensor):
        A = A.reshape(A.shape[0],A.shape[1],3,3)
        B = B.reshape(A.shape[0],A.shape[1],3,3)
        result = torch.einsum('...ik,...kj->...ij', A, B)
        result = result.reshape(result.shape[0],result.shape[1],9)
        return result


    def __transpose(self, A: torch.Tensor):
        # Get the shape of the input tensor
        shape = A.shape

        # Reshape the tensor to a matrix of shape (N, 3, 3)
        reshaped_A = A.reshape(-1, 3, 3)

        # Transpose the matrix along the last two dimensions
        transposed_A = reshaped_A.transpose(-2, -1)

        # Reshape the transposed matrix back to a vector
        transposed_A = transposed_A.reshape(shape)

        return transposed_A.clone()

    def __trace(self,A: torch.Tensor):
        A = A.reshape(A.shape[0],A.shape[1],9)
        ret = A[:,:,::4].sum(dim=-1)
        return ret
    
    def __ELAS_ddot_A(self, A: torch.Tensor, crystal: Crystal):
        ret = A.clone()
        ELAS = torch.tensor(crystal.ELAS, device=self.device)
        ELAS_expanded = torch.zeros((A.shape[0],A.shape[1],9,9),device=self.device,dtype=torch.float64)
        ELAS_expanded[:,:,:6,:6] = ELAS
        ret[:,:] = torch.einsum('...ik,...k->...i', ELAS_expanded[:,:], ret[:,:])

        ret[:, :, -3:] = ret[:, :, -6:-3]
        return ret





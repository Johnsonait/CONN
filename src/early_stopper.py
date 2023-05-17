import numpy as np

class EarlyStopper():

    def __init__(self,patience = 50, delta = 1e-4):
        self.patience = patience
        self.delta = delta
        self.min_loss = np.inf
        self.count = 0
    
    def should_stop(self,loss):
        if loss >= self.min_loss:
            self.count += 1
            if self.count >= self.patience:
                return True
        else:
            self.count = 0
            self.min_loss = loss
            return False

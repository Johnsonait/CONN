import torch
from torch.optim import Adam
from torch.nn import MSELoss

from src.conn.conn_dataset import CONNDataset
from src.conn.crystal_conn import CrystalCONN
from src.trainer import Trainer
from src.run_model import ModelTester

def main():
    device = torch.device('cpu')

    training_dataset = CONNDataset(split='training',device=device)
    validation_dataset = CONNDataset(split='validation',device=device)

    model = CrystalCONN(
                            hidden_size=128,
                            loss_fcn=MSELoss,
                            device=device
                        )

    trainer = Trainer(
                        model=model,
                        train_dataset=training_dataset,
                        val_dataset=validation_dataset,
                        optimizer=Adam,
                        epochs = 1000,
                        batch_size= 64,
                        train_params={'lr': 1e-4},
                        device=device
                    )

    trainer.fit()

    tester = ModelTester(model=model, dataset=validation_dataset)


    return

if __name__ == '__main__':
    main()
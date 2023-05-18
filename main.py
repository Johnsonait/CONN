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

    model_params = {
        'hidden_size': 8,
        'loss_fcn': MSELoss,
        'device': device
    }
    model = CrystalCONN(**model_params)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable Parameters: {pytorch_total_params:.4e}')

    trainer = Trainer(
                        model=model,
                        train_dataset=training_dataset,
                        val_dataset=validation_dataset,
                        optimizer=Adam,
                        epochs = 1000,
                        batch_size= 256,
                        train_params={'lr': 1e-4},
                        device=device
                    )

    train = False
    if train: 
        trainer.fit()
    else:
        model = model.load(model_params,'./model.pt')
        tester = ModelTester(model=model,dataset=validation_dataset)
        tester.run_model()

    return

if __name__ == '__main__':
    main()
import torch
from torch.optim import Adam
from torch.nn import MSELoss

from src.conn.conn_dataset import CONNDataset
from src.conn.crystal_conn import CrystalCONN
from src.trainer import Trainer
from src.run_model import ModelTester



def main():
    device = torch.device('cpu')

    model_params = {
        'hidden_size': 2048,
        'loss_fcn': MSELoss,
        'device': device
    }
    model = CrystalCONN(**model_params)

    # Count the total number of trainable parameters for the mode. Useful for
    # comparing approaches/architectures
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable Parameters: {pytorch_total_params:.4e}')

    torch.manual_seed(0)
    train = False
    if train: 
        training_dataset = CONNDataset(split='training',device=device)
        validation_dataset = CONNDataset(split='validation',device=device)

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
    else:
        model = model.load(model_params,'./model.pt')
        validation_dataset = CONNDataset(split='validation',device=device)
        tester = ModelTester(model=model,dataset=validation_dataset)
        tester.run_model()
        tester.show()

    return

if __name__ == '__main__':
    main()

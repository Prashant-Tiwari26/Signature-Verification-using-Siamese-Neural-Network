import torch
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append(os.getcwd())
from utils.models import shufflenet_model
from utils.data import SiameseDataset
from utils.train import ContrastiveLoss, train_loop

def train_model():
    model = shufflenet_model()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
    def lr_lambda(epoch):
        if epoch <= 15:
            return 0.85
        return 1
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)

    train_data = SiameseDataset('Data/custom/train.csv', 'Data/custom/full', 224)
    val_data = SiameseDataset('Data/custom/val.csv', 'Data/custom/full', 224, False)

    TrainDataloader = DataLoader(train_data, batch_size=64)
    ValDataloader = DataLoader(val_data, batch_size=64)

    print("No. of batches in training set = {}".format(len(TrainDataloader)))
    print("No. of batches in validation set = {}\n\n".format(len(ValDataloader)))
    
    train_loop(model, optimizer, ContrastiveLoss(), scheduler, TrainDataloader, ValDataloader, 'Data/custom/Performance/training_effb0.png', 30, 100)

    model_path = "Models/shufflenet.pth"
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    train_model()
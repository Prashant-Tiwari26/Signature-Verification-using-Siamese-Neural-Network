import torch
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append(os.getcwd())
from utils.models import efficientnet_model
from utils.data import SiameseDataset
from utils.train import ContrastiveLoss, train_loop

def train_model():
    model = efficientnet_model()
    optimizer_bn = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
    def lr_lambda(epoch):
        if epoch <= 15:
            return 0.85
        return 1
    scheduler_bn = torch.optim.lr_scheduler.MultiplicativeLR(optimizer_bn, lr_lambda)

    train_data = SiameseDataset('Data/custom/train.csv', 'Data/custom/full', 224)
    val_data = SiameseDataset('Data/custom/val.csv', 'Data/custom/full', 224, False)

    TrainDataloader = DataLoader(train_data, batch_size=32)
    ValDataloader = DataLoader(val_data, batch_size=32)

    print("No. of batches in training set = {}".format(len(TrainDataloader)))
    print("No. of batches in validation set = {}\n\n".format(len(ValDataloader)))
    
    train_loop(model, optimizer_bn, ContrastiveLoss(), scheduler_bn, TrainDataloader, ValDataloader, 'Data/custom/Performance/training_effb0.png', 30, 25)

    model_path = "Models/efficientb0.pth"
    torch.save(model, model_path)

if __name__ == '__main__':
    train_model()
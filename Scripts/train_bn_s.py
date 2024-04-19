"""
Script for training a Siamese Neural Network model for signature verification.

This script trains a Siamese Neural Network model for signature verification using
the specified dataset and hyperparameters. It utilizes the `models.Model_BN_s` model
architecture and the `ContrastiveLoss` loss function for training. The trained model
is saved to a specified file.

Usage:
- Ensure that you have the required dataset CSV files in the 'Data/custom/' directory:
  - 'train_data.csv' for training data
  - 'val_data.csv' for validation data
- Execute this script to train the model and save it to 'Models/bn_s.pth'.

Note:
Ensure that the necessary Python packages and modules are available, and the paths to
the dataset and model are correctly configured.
"""
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.append(os.getcwd())
from utils.models import Model_BN_s
from utils.data import SiameseDataset
from utils.train import ContrastiveLoss, train_loop

def TrainModel():
    bn_model_custom = Model_BN_s()

    optimizer_bn = torch.optim.AdamW(params=bn_model_custom.parameters(), lr=1e-4)
    def lr_lambda(epoch):
        if epoch <= 15:
            return 0.85
        return 1
    scheduler_bn = torch.optim.lr_scheduler.MultiplicativeLR(optimizer_bn, lr_lambda)

    train_data = SiameseDataset('Data/custom/train.csv', 'Data/custom/full', [155,200])
    val_data = SiameseDataset('Data/custom/val.csv', 'Data/custom/full', [155,200], False)

    TrainDataloader = DataLoader(train_data, batch_size=32)
    ValDataloader = DataLoader(val_data, batch_size=32)

    print("No. of batches in training set = {}".format(len(TrainDataloader)))
    print("No. of batches in validation set = {}\n\n".format(len(ValDataloader)))
    
    train_loop(bn_model_custom, optimizer_bn, ContrastiveLoss(), scheduler_bn, TrainDataloader, ValDataloader, 'Data/custom/Performance/training.png', 30, 25)

    model_path = "Models/bn_s.pth"
    torch.save(bn_model_custom, model_path)

if __name__ == '__main__':
    TrainModel()
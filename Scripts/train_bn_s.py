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

import sys
sys.path.append("C:\College\Projects\Signature Verification using Siamese Neural Network")
from utils.models import Model_BN_s
from utils.data import SiameseDataset
from utils.train import ContrastiveLoss, transform, TrainLoopV2

def TrainModel():
    bn_model_custom = Model_BN_s()

    optimizer_bn = torch.optim.NAdam(params=bn_model_custom.parameters(), lr=1e-4, weight_decay=0.0005)

    scheduler_bn = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_bn, step_size=10, gamma=0.1)

    train_data = SiameseDataset('Data/custom/used data/train.csv', 'Data/custom/full', transforms=transform)
    val_data = SiameseDataset('Data/custom/used data/val.csv', 'Data/custom/full', transforms=transform)

    TrainDataloader = DataLoader(train_data, batch_size=32)
    ValDataloader = DataLoader(val_data, batch_size=32)
    
    TrainLoopV2(bn_model_custom, optimizer_bn, ContrastiveLoss(), 20, scheduler_bn, TrainDataloader, ValDataloader, 5)

    model_path = "Models/bn_s.pth"
    torch.save(bn_model_custom, model_path)

if __name__ == '__main__':
    TrainModel()
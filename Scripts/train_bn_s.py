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

import sys
sys.path.append("C:\College\Projects\Signature Verification using Siamese Neural Network")
from Models.models import Model_BN_s
from utils import ContrastiveLoss, transform, SiameseDataset, TrainLoopV2

def TrainModel():
    bn_model_custom = Model_BN_s()

    optimizer_bn = torch.optim.RMSprop(params=bn_model_custom.parameters(), lr=1e-4, weight_decay=0.0005, momentum=0.9, eps=1e-8)

    scheduler_bn = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_bn, step_size=10, gamma=0.1)

    train_data = SiameseDataset('Data/custom/train_data.csv', 'Data/custom/full', transforms=transform)
    val_data = SiameseDataset('Data/custom/val_data.csv', 'Data/custom/full', transforms=transform)

    TrainDataloader = DataLoader(train_data, batch_size=32)
    ValDataloader = DataLoader(val_data, batch_size=32)

    TrainLoopV2(bn_model_custom, optimizer_bn, ContrastiveLoss(), 20, scheduler_bn, TrainDataloader, ValDataloader, 5)

    model_path = "Models/bn_s.pth"
    torch.save(Model_BN_s, model_path)

if __name__ == '__main__':
    TrainModel()
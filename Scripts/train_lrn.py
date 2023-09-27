import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("C:\College\Projects\Signature Verification using Siamese Neural Network")
from Models.models import Model_LRN
from utils import ContrastiveLoss, transform, SiameseDataset, TrainLoop

lrn_model_custom = Model_LRN()

optimizer_lrn = torch.optim.RMSprop(params=lrn_model_custom.parameters(), lr=1e-4, weight_decay=0.0005, momentum=0.9, eps=1e-8)

scheduler_lrn = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_lrn, step_size=10, gamma=0.1)

train_data = SiameseDataset('Data/custom/train_data.csv', 'Data/custom/full', transforms=transform)
val_data = SiameseDataset('Data/custom/val_data.csv', 'Data/custom/full', transforms=transform)

TrainDataloader = DataLoader(train_data, batch_size=64)
ValDataloader = DataLoader(val_data, batch_size=64)

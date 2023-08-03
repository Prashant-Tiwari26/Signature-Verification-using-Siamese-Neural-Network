import torch
from torch.utils.data import DataLoader
from model import Model_BN, Model_LRN
from utils import ContrastiveLoss, transform, SiameseDataset, TrainLoop

bn_model_custom = Model_BN()
lrn_model_custom = Model_LRN()

optimizer_bn = torch.optim.RMSprop(params=bn_model_custom.parameters(), lr=1e-4, weight_decay=0.0005, momentum=0.9, eps=1e-8)
optimizer_lrn = torch.optim.RMSprop(params=lrn_model_custom.parameters(), lr=1e-4, weight_decay=0.0005, momentum=0.9, eps=1e-8)

scheduler_bn = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_bn, step_size=10, gamma=0.1)
scheduler_lrn = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_lrn, step_size=10, gamma=0.1)

train_data = SiameseDataset('Data/custom/train_val.csv', 'Data/custom/full', transforms=transform)
test_data = SiameseDataset('Data/custom/test.csv', 'Data/custom/full', transforms=transform)

TrainDataloader = DataLoader(train_data, batch_size=64)
TestDataloader = DataLoader(test_data, batch_size=64)

TrainLoop(bn_model_custom, optimizer_bn, ContrastiveLoss(), 20, scheduler_bn, TrainDataloader, TestDataloader)
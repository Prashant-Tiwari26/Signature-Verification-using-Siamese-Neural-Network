import torch
from model import Model_BN, Model_LRN

bn_model = Model_BN()
lrn_model = Model_LRN()

optimizer_bn = torch.optim.RMSprop(params=bn_model.parameters(), lr=1e-4, weight_decay=0.0005, momentum=0.9, eps=1e-8)
optimizer_lrn = torch.optim.RMSprop(params=lrn_model.parameters(), lr=1e-4, weight_decay=0.0005, momentum=0.9, eps=1e-8)

scheduler_bn = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_bn, step_size=10, gamma=0.1)
scheduler_lrn = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_lrn, step_size=10, gamma=0.1) 
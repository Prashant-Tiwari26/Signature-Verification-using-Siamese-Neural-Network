import os
import sys
import json
import torch
from sklearn import metrics

sys.path.append(os.getcwd())
from utils.data import SiameseDataset
from utils.models import shufflenet_model
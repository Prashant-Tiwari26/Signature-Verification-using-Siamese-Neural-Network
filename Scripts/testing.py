"""
Signature Verification Prediction Script

This script performs signature verification using a trained Siamese Neural Network model. It loads a pre-trained model, processes test data, computes similarity scores, and generates predictions.

Requirements:
- PyTorch
- NumPy
- The project structure should include a 'utils' module with 'transform' and 'SiameseDataset' classes,
  and a 'Models' module with the 'Model_BN_s' class.

Usage:
- Ensure that the trained model file 'Models/bn_s.pth' and the test dataset 'Data/custom/used data/test.csv' exist.
- Modify the paths and settings as needed to match your project configuration.
- Run the script to perform prediction and save the results.
"""
import sys
import torch
import warnings
import numpy as np
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

sys.path.append("C:\College\Projects\Signature Verification using Siamese Neural Network")
from utils import transform, SiameseDataset
from utils.models import Model_BN_s

def Predict():
    data = SiameseDataset("Data/custom/used data/test.csv", "Data/custom/full", transforms=transform)
    TestDataloader = DataLoader(dataset=data, batch_size=32)

    model = Model_BN_s()
    model.load_state_dict(torch.load("Models/bn_s.pth").state_dict())
    model.to("cuda")
    distances = []
    model.eval()
    with torch.inference_mode():
        for x1, x2, y in TestDataloader:
            x1 = x1.to("cuda")
            x2 = x2.to("cuda")
            outputs = model(x1, x2)
            distances.extend(outputs.cpu().numpy())

    distances = np.array(distances)

    np.save("Data/custom/Performance/test_performance.npy", distances)

    preds = distances.copy()
    for i in range(len(preds)):
        if preds[i]>0.5:
            preds[i] = 0
        else:
            preds[i] = 1

    np.save("Data/custom/Performance/test_predictions.npy", preds)

if __name__ == '__main__':
    Predict()
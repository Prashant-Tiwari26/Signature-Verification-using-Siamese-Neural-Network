"""
Signature Verification Model Evaluation Script

This script evaluates the performance of a signature verification model using test data. It calculates various evaluation metrics including accuracy, classification report, confusion matrix, and ROC curve with AUC score.

Requirements:
- NumPy
- pandas
- Matplotlib
- Scikit-learn
- PyTorch (if used for training)

Usage:
- Ensure that the test data file 'Data/custom/used data/test.csv' and the predicted labels file 'Data/custom/Performance/test_predictions.npy' exist.
- Modify the paths and settings as needed to match your project configuration.
- Run the script to evaluate the model and visualize the results.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

def Evaluate():
    test_data = pd.read_csv("Data/custom/used data/test.csv")
    true_labels = np.array(test_data['2'])
    pred_labels = np.load("Data/custom/Performance/test_predictions.npy")

    print("Accuracy = {}%\n".format(accuracy_score(true_labels, pred_labels)*100))
    print(print(classification_report(true_labels, pred_labels)))

    cf = confusion_matrix(true_labels, pred_labels)
    matrix_disp = ConfusionMatrixDisplay(cf)
    matrix_disp.plot(cmap='Reds')

    fpr, tpr, thresholds = roc_curve(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='orange', linewidth=5)
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    Evaluate()
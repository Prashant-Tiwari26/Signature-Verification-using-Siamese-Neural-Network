"""
Script for splitting a dataset into training, validation, and test sets and saving them as CSV files.

This script reads a CSV file named 'data.csv' from the 'Data/custom/' directory,
performs a series of data splits, and saves the resulting splits as CSV files.
The splits are stratified, ensuring that the distribution of the target variable '2'
is maintained in each split.

The data splits are as follows:
- The original dataset is split into 'train' and 'test' sets with an 80/20 ratio.
- The 'train' set is further split into 'train', 'validation', and 'unused' sets
  with an 80/10/10 ratio.
- A subset of the 'train' set called 'used' is created, and it is split into
  'train', 'validation', and 'test' sets with an 80/10/10 ratio.

The resulting splits are saved as CSV files:
- 'train_data.csv': Original training data split.
- 'test_data.csv': Original test data split.
- 'val_data.csv': Original validation data split.
- 'train.csv': Used training data split.
- 'test.csv': Used test data split.
- 'val.csv': Used validation data split.

Usage:
- Place a CSV file named 'data.csv' in the 'Data/custom/' directory with the
  target variable '2'.
- Execute this script to perform the data splits and save the resulting CSV files.

Note:
Ensure that the required libraries (pandas and scikit-learn) are installed
to run this script successfully.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def SplitData():
    data = pd.read_csv("Data/custom/data.csv")

    train, test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data['2'])
    train, val = train_test_split(train, test_size=0.125, shuffle=True, stratify=train['2'])

    train.to_csv("Data/custom/train_data.csv", index=False)
    test.to_csv("Data/custom/test_data.csv", index=False)
    val.to_csv("Data/custom/val_data.csv", index=False)

    used, unused = train_test_split(train, test_size=0.5, shuffle=True, stratify=data['2'])

    train_used, test_used = train_test_split(used, test_size=0.2, shuffle=True, stratify=used['2'])
    train_used, val_used = train_test_split(train_used, test_size=0.125, shuffle=True, stratify=train_used['2'])

    train_used.to_csv("Data/custom/used data/train.csv", index=False)
    test_used.to_csv("Data/custom/used data/test.csv", index=False)
    val_used.to_csv("Data/custom/used data/val.csv", index=False)

if __name__ == '__main__':
    SplitData()
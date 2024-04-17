"""
Script for splitting a dataset into training, validation, and test sets and saving them as CSV files.

This script reads a CSV file named 'data.csv' from the 'Data/custom/' directory,
performs a series of data splits, and saves the resulting splits as CSV files.
The splits are stratified, ensuring that the distribution of the target variable '2'
is maintained in each split.

The data splits are as follows:
- Person column is added to original dataset which will be used to make sure that training, testing
  and validation sets have signatures of different people with no overlap.
- Dataset is split into 2 datasets with all authentic matches in df_authentic and forgeries in df_forgery.
- The datasets are then split into 'train', 'validation' sets with numbers selected such that approximately
  75% data goes to training set, 8% to validation and rest to testing.
- The 'person' column is dropped and datasets are shuffled before being saved.

The resulting splits are saved as CSV files:
- 'train_data.csv': Original training data split.
- 'test_data.csv': Original test data split.
- 'val_data.csv': Original validation data split.

Usage:
- Place a CSV file named 'data.csv' in the 'Data/custom/' directory with the
  target variable '2'.
- Execute this script to perform the data splits and save the resulting CSV files.

Note:
Ensure that the required libraries (pandas and scikit-learn) are installed
to run this script successfully.
"""

import pandas as pd

def SplitData():
    data = pd.read_csv("Data/custom/data.csv", index_col=False)

    data['person'] = data['0'].apply(lambda x: x[:3])
    df_authentic = data.loc[data['2'] == 1].copy()
    df_forgery = data.loc[data['2'] == 0].copy()
    persons = list(data['person'].unique())
    del data

    train = pd.DataFrame(columns=['0', '1', '2', 'person'])
    val = pd.DataFrame(columns=['0', '1', '2', 'person'])
    test = pd.DataFrame(columns=['0', '1', '2', 'person'])

    for person in persons:
        if len(train) < 63800:
            train = pd.concat([train, df_authentic[df_authentic['person'] == person]], axis=0)
            train = pd.concat([train, df_forgery[df_forgery['person'] == person]], axis=0)
        elif len(test) < 14650:
            test = pd.concat([test, df_authentic[df_authentic['person'] == person]], axis=0)
            test = pd.concat([test, df_forgery[df_forgery['person'] == person]], axis=0)
        else:
            val = pd.concat([val, df_authentic[df_authentic['person'] == person]], axis=0)
            val = pd.concat([val, df_forgery[df_forgery['person'] == person]], axis=0)

    del df_authentic, df_forgery

    print("Rows in training data = {}".format(len(train)))
    print("Rows in testing data = {}".format(len(test)))
    print("Rows in validation data = {}".format(len(val)))

    print("Label distribution for training set :\n{}".format(train['2'].value_counts()))
    print("Label distribution for testing set :\n{}".format(test['2'].value_counts()))
    print("Label distribution for validation set :\n{}".format(val['2'].value_counts()))

    train.drop(['person'], axis=1, inplace=True)
    test.drop(['person'], axis=1, inplace=True)
    val.drop(['person'], axis=1, inplace=True)

    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    val = val.sample(frac=1).reset_index(drop=True)

    train.to_csv("Data/custom/train.csv", index=False)
    test.to_csv("Data/custom/test.csv", index=False)
    val.to_csv("Data/custom/val.csv", index=False)

if __name__ == '__main__':
    SplitData()
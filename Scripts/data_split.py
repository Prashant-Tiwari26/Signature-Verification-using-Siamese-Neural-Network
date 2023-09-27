import pandas as pd
from sklearn.model_selection import train_test_split

def SplitData():
    data = pd.read_csv("Data/custom/data.csv")

    train, test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data['2'])
    train, val = train_test_split(train, test_size=0.125, shuffle=True, stratify=train['2'])

    train.to_csv("Data/custom/train_data.csv", index=False)
    test.to_csv("Data/custom/test_data.csv", index=False)
    val.to_csv("Data/custom/val_data.csv", index=False)

if __name__ == '__main__':
    SplitData()
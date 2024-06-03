# Signature Verification using Siamese Neural Network

## Overview

This project is an implementation of an smaller model that is inspired from model SigNet given in the paper:
[*Dey, S., Dutta, A., Toledo, J. I., Ghosh, S. K., Lladós, J., & Pal, U. (2017). Signet: Convolutional siamese network for writer independent offline signature verification. arXiv preprint arXiv:1707.02131.*](https://arxiv.org/pdf/1707.02131.pdf) It uses siamese neural network structure with two parallel CNNs to verify if a signature is a forgery or not.

## Table of Contents
+ Overview
+ Table of Contents
+ Datasets
+ Model Architecture
+ Preprocessing
+ Training
+ Evaluation
+ Usage
+ Dependencies

## Datasets

The following datasets have been using in training and evaluation of the model.

### CEDAR

CEDAR signature database contains signatures of 55 signers belonging to various cultural and professional backgrounds. Each of these signers signed 24 genuine signatures 20 minutes apart. Each of the forgers tried to emulate the signatures of 3 persons, 8 times each, to produce 24 forged signatures for each of the genuine signers. Hence the dataset comprise 55 × 24 = 1320 genuine signatures as well as 1320 forged signatures.
<br><br>
Link for [*Dataset*](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets)

### Custom Dataset

This dataset contains the signature of 124 users both genuine and fraud signature for signature verification. Each person has around 10 Genuine signatures which they made themselves and around 10 Forged signatures someone else made. All the data is extracted from ICDAR 2011 Signature Dataset and CEDAR Signature Verification Dataset.
<br><br>
Link for [*Dataset*](https://www.kaggle.com/datasets/mallapraveen/signature-matching)

### Data used for training

Due to limited computation power I have trimmed the custom dataset down to 40% of its original size and then used it for training, validation and testing purposes.

## Model Architecture

### For SigNet
The Model Architecture is as follows:<br><br>
Model_LRN(<br>
  (model_branch): Sequential(<br>
    (0): Conv2d(3, 96, kernel_size=(11, 11), stride=(1, 1))<br>
    (1): ReLU()<br>
    (2): LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2)<br>
    (3): ReLU()<br>
    (4): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)<br>
    (5): Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))<br>
    (6): ReLU()<br>
    (7): LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),<br>
    (8): ReLU()<br>
    (9): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)<br>
    (10): Dropout2d(p=0.3, inplace=False)<br>
    (11): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
    (12): ReLU()<br>
    (13): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
    (14): ReLU()<br>
    (15): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)<br>
    (16): Dropout2d(p=0.3, inplace=False)<br>
    (17): Flatten(start_dim=1, end_dim=-1)<br>
    (18): Linear(in_features=108800, out_features=1024, bias=True)<br>
    (19): ReLU()<br>
    (20): Dropout1d(p=0.5, inplace=False)<br>
    (21): Linear(in_features=1024, out_features=128, bias=True)<br>
  )<br>
)<br>

Total params: 113,963,840<br>
Trainable params: 113,963,840<br>
Non-trainable params: 0<br>

### For the model wit batch normalization
The Model Architecture is as follows:<br><br>
Model_BN(<br>
  (model_branch): Sequential(<br>
    (0): Conv2d(3, 96, kernel_size=(11, 11), stride=(1, 1))<br>
    (1): SELU()<br>
    (2): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
    (3): SELU()<br>
    (4): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)<br>
    (5): Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))<br>
    (6): SELU()<br>
    (7): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),<br>
    (8): SELU()<br>
    (9): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)<br>
    (10): Dropout2d(p=0.3, inplace=False)<br>
    (11): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
    (12): SELU()<br>
    (13): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
    (14): SELU()<br>
    (15): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)<br>
    (16): Dropout2d(p=0.3, inplace=False)<br>
    (17): Flatten(start_dim=1, end_dim=-1)<br>
    (18): Linear(in_features=108800, out_features=1024, bias=True)<br>
    (19): SELU()<br>
    (20): Dropout1d(p=0.5, inplace=False)<br>
    (21): Linear(in_features=1024, out_features=128, bias=True)<br>
  )<br>
)<br>

Total params: 113,963,840<br>
Trainable params: 113,963,840<br>
Non-trainable params: 0<br>


### For smaller model
The Model Architecture is as follows:<br><br>
Model_BN_s(<br>
  (model_branch): Sequential(<br>
    (0): Conv2d(3, 96, kernel_size=(11, 11), stride=(1, 1))<br>
    (1): SELU()<br>
    (2): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)<br>
    (3): SELU()<br>
    (4): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)<br>
    (5): Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))<br>
    (6): SELU()<br>
    (7): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),<br>
    (8): SELU()<br>
    (9): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)<br>
    (10): Dropout2d(p=0.3, inplace=False)<br>
    (11): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
    (12): SELU()<br>
    (13): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
    (14): SELU()<br>
    (15): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)<br>
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
    (17): SELU()<br>
    (18): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)<br>
    (19): Dropout2d(p=0.3, inplace=False)<br>
    (20): Flatten(start_dim=1, end_dim=-1)<br>
    (21): Linear(in_features=20480, out_features=1024, bias=True)<br>
    (22): SELU()<br>
    (23): Dropout1d(p=0.5, inplace=False)<br>
    (24): Linear(in_features=1024, out_features=128, bias=True)<br>
  )<br>
)<br>

Total params: 28,285,312<br>
Trainable params: 28,285,312<br>
Non-trainable params: 0<br>

## Preprocessing

* The custom class for dataset to be used for this project has been given `utils.py` by the name *SiameseDataset*.
* The transform function is added into the class as a function that converts image to torch tensors and resizes them.
* If its part of train set then for data augmentation purposes randomized horizontal and vertical flips are also added.
## Training

* The model has been trained on a trimmed down dataset of 20,884 entries with a validation set of 2984 entries.
* The architecture of models is given in `src/models/cnn.py`
* The code for Model trainer class is given in `src/models/train.py`
* Model has been trained using *AdamW optimizer* for 30 epochs with an initial learning rate of 3e-4 and early if loss does not improve for 6 epochs.
* For the first 15 epochs the learning rate reduces 15% per epoch.
* Custom Contrastive Loss function has been used which is availabe in `src/utils/loss.py`

## Evaluation

* The model has been tested on testing data by setting a euclidean distance threshold value of 0.15 for smaller version to model mentioned in original paper, 0.242 for shufflenet, below it the signatures are authentic, above it there is a forgery. The code for this task is given in `src/models/evaluate.py`

### The Confusion matrices 
![reports\figures\custom_cf.png](reports\figures\custom_cf.png)
![reports\figures\shufflenet_cf.png](reports\figures\shufflenet_cf.png)

### Classification Report

```
For shufflenet
{
    "0": {
        "precision": 0.9259771705292287,
        "recall": 0.945268361581921,
        "f1-score": 0.9355233269264371,
        "support": 2832.0
    },
    "1": {
        "precision": 0.942099364960777,
        "recall": 0.9217836257309941,
        "f1-score": 0.9318307777572511,
        "support": 2736.0
    },
    "accuracy": 0.9337284482758621,
    "macro avg": {
        "precision": 0.9340382677450028,
        "recall": 0.9335259936564575,
        "f1-score": 0.9336770523418441,
        "support": 5568.0
    },
    "weighted avg": {
        "precision": 0.9338992833102481,
        "recall": 0.9337284482758621,
        "f1-score": 0.9337088846622681,
        "support": 5568.0
    }
}
```

```
For custom model
{
    "0": {
        "precision": 0.9811937857726901,
        "recall": 0.847457627118644,
        "f1-score": 0.9094353921940129,
        "support": 2832.0
    },
    "1": {
        "precision": 0.8616271620755925,
        "recall": 0.9831871345029239,
        "f1-score": 0.9184021850460908,
        "support": 2736.0
    },
    "accuracy": 0.9141522988505747,
    "macro avg": {
        "precision": 0.9214104739241413,
        "recall": 0.915322380810784,
        "f1-score": 0.9139187886200519,
        "support": 5568.0
    },
    "weighted avg": {
        "precision": 0.9224412206801508,
        "recall": 0.9141522988505747,
        "f1-score": 0.9138414886816719,
        "support": 5568.0
    }
}
```

Shuffle Net model performs slightly better with much lower number of parameters(2.6 million compared to 28 million).
> The training and testing on other datasets will be done later on.

## Usage

The project can be run in three different ways:
* To train only: ```python .\ train.yaml```
* To evaluate only: ```python .\ eval.yaml```
* To the do both as a pipeline: ```python .\ train.yaml eval.yaml```

Adjust values of parameters in train  and eval config files accordingly. 

## Dependencies

All dependencies can be installed by running the following command in the terminal:

```pip install -r requirements.txt```
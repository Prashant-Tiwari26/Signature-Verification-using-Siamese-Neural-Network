import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from timeit import default_timer as timer
from PIL import Image
import matplotlib.pyplot as plt
from PIL.ImageOps import invert
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode

transform = transforms.Compose([
    invert,
    ToTensor(),
    Resize((155,220), interpolation=InterpolationMode.BICUBIC),
    Normalize(mean=0.5, std=0.5)
])
    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin: int = 1) -> None:
        """
        Contrastive Loss function for training a neural network with pairwise distance-based contrastive loss.

        Args:
            margin (int, optional): The margin for the contrastive loss. Default is 1.

        Note:
            The contrastive loss function aims to minimize the distance between embeddings of similar pairs
            and maximize the distance between embeddings of dissimilar pairs.

            The formula for the contrastive loss is:
            loss = y * (dist^2) + (1 - y) * max(margin - dist, 0)^2

            where:
            - dist: The Euclidean distance between two embeddings.
            - y: The binary label (1 for similar pairs, 0 for dissimilar pairs).
            - margin: The margin for the contrastive loss.

        Example:
            loss = ContrastiveLoss(margin=1)
            embedding1 = torch.tensor([1.0, 2.0])
            embedding2 = torch.tensor([3.0, 4.0])
            similarity_label = 1  # Similar pair
            output = loss(torch.norm(embedding1 - embedding2), similarity_label)
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, y):
        """
        Forward pass of the contrastive loss function.

        Args:
            dist (torch.Tensor): The Euclidean distance between two embeddings.
            y (torch.Tensor): The binary label indicating the similarity between the embeddings (1 for similar pairs, 0 for dissimilar pairs).

        Returns:
            torch.Tensor: The contrastive loss value.

        Note:
            This function computes the contrastive loss for a pair of embeddings based on their distance and similarity
            label. The loss aims to pull similar pairs closer together and push dissimilar pairs apart in the embedding space.
        """
        loss = y * torch.pow(dist, 2) + (1 - y) * torch.pow(torch.clamp(self.margin - dist, min=0), 2)
        return loss.mean()
    
class SiameseDataset(Dataset):
    """
    Custom PyTorch Dataset for Siamese neural network training on image pairs.

    This dataset loads pairs of grayscale images and their corresponding labels from a CSV file. 
    The CSV file should contain image file names of two files in each row along with a label 
    (1 if the images are similar, 0 if they are not). The images are loaded from the specified 
    'image_dir' directory.

    Parameters:
        labels_csv (str): Path to the CSV file containing image pairs and their labels.
        image_dir (str): Path to the directory containing the images.
        transforms (transforms.Compose, optional): A composition of PyTorch transforms to be applied 
            to the images. Default is None.

    Returns:
        tuple: A tuple containing two images (image1 and image2) and their corresponding label.
            - image1 (torch.Tensor): The first grayscale image, converted to a PyTorch tensor.
            - image2 (torch.Tensor): The second grayscale image, converted to a PyTorch tensor.
            - label (torch.Tensor): The label indicating whether the images are similar (1) or not (0).
    
    Note:
        - The CSV file should have three columns: 'image1', 'image2', and 'label'.
        - The 'transforms' argument is an optional parameter that allows applying transformations 
          to the images. The transformations should be a composition of transforms from the 
          'torchvision.transforms' module. If not provided, the images will be returned as PIL Images.
        - The images are loaded as grayscale ('L') images.

    Example:
        # Assuming you have a CSV file 'data.csv' and a folder 'images' containing the images
        # Initialize the dataset with default transforms
        data = SiameseDataset("data.csv", "images")

        # Or, initialize the dataset with custom transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        data = SiameseDataset("data.csv", "images", transform=transform)
    """
    def __init__(self, labels_csv:str, image_dir:str, transforms:transforms=None) -> None:
        super().__init__()
        self.labels = pd.read_csv(labels_csv, index_col=False)
        self.image_dir = image_dir
        self.transform = transforms

    def __getitem__(self, index):
        image1_path = os.path.join(self.image_dir, str(self.labels.iat[index, 0]))
        image2_path = os.path.join(self.image_dir, str(self.labels.iat[index, 1]))
        label = torch.tensor(self.labels.iat[index,2])
        
        image1 = Image.open(image1_path).convert("L")
        image2 = Image.open(image2_path).convert("L")

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)        

        return image1, image2, label
    
    def __len__(self):
        return len(self.labels)
    
def TrainLoop(
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        loss_function:torch.nn.Module,
        num_epochs:int,
        scheduler:torch.optim.lr_scheduler.StepLR,
        train_dataloader:torch.utils.data.DataLoader,
        early_stopping_rounds:int,
        test_dataloader:torch.utils.data.DataLoader=None,
        val_dataloader:torch.utils.data.DataLoader=None,
        device:str='cuda'
):
    """
    TrainLoop function for training a PyTorch neural network model using the specified data and settings.

    Parameters:
        model (torch.nn.Module): The PyTorch neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters during training.
        loss_function (torch.nn.Module): The loss function used to compute the training and validation loss.
        num_epochs (int): The number of epochs to train the model for.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler for the optimizer.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset (used for early stopping).
        early_stopping_rounds (int): Number of epochs to wait before stopping training if there is no improvement in validation loss.
        val_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the validation dataset. Default is None.
        device (str, optional): Device to use for training (e.g., 'cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        None: This function does not return any value. It trains the model and prints the progress.

    Note:
        - The model, optimizer, loss function, and scheduler should be properly initialized before calling this function.
        - The train and validation dataloaders should provide batches of data in the format (x1, x2, y), where x1 and x2 are input tensors and y is the target tensor.
        - The test dataloader is used for early stopping based on validation loss. If early stopping is not required, set `test_dataloader` to None.
        - The model will be moved to the specified `device` before training.
        - This function uses tqdm for displaying the training progress.
    """
    model.to(device)
    start_time = timer()
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_loss = 0
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch : {epoch}\n----------------------")
        for batch, (x1, x2, y) in enumerate(train_dataloader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            distance = model(x1, x2)
            loss = loss_function(distance, y)
            loss.backward()
            optimizer.step()
            print(f"Loss for batch: {batch} = {loss}")
            train_loss += loss

        print(f"Training Loss = {train_loss}")

        if val_dataloader is not None:
            model.eval()
            validation_loss = 0
            with torch.inference_mode():
                for x1, x2, y in val_dataloader:
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    y = y.to(device)
                    distance = model(x1,x2)
                    loss = loss_function(distance, y)
                    validation_loss+=loss

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement+=1

                print(f"Current Validation Loss = {validation_loss}")
                print(f"Best Validation Loss = {best_val_loss}")
                print(f"Epochs without Improvement = {epochs_without_improvement}")

                if epochs_without_improvement > early_stopping_rounds:
                    print("Early Stoppping Triggered")
                    break

        scheduler.step()

    end_time = timer()
    print(f"Training Time = {end_time-start_time} seconds")

def TrainLoopV2(
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        loss_function:torch.nn.Module,
        num_epochs:int,
        scheduler:torch.optim.lr_scheduler.StepLR,
        train_dataloader:torch.utils.data.DataLoader,
        val_dataloader:torch.utils.data.DataLoader,
        early_stopping_rounds:int,
        device:str='cuda',
        return_best_model:bool=True
):
    """
    Training loop for a deep learning model with optional early stopping.

    Args:
        model (torch.nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loss_function (torch.nn.Module): The loss function used for training.
        num_epochs (int): The number of training epochs.
        scheduler (torch.optim.lr_scheduler.StepLR): Learning rate scheduler.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        early_stopping_rounds (int): Number of consecutive epochs without improvement on the validation loss to trigger early stopping.
        device (str, optional): Device to use for training (e.g., 'cuda' or 'cpu'). Default is 'cuda'.
        return_best_model (bool, optional): Whether to return the model with the best validation loss. Default is True.

    Returns:
        None: If `return_best_model` is False.
        torch.nn.Module: The trained model with the best validation loss if `return_best_model` is True.
    
    The function trains the specified `model` using the provided `optimizer` and
    `loss_function` for the specified number of `num_epochs`. It monitors the
    validation loss and performs early stopping if the validation loss does not
    improve for a specified number of consecutive epochs (controlled by
    `early_stopping_rounds`). The best model (lowest validation loss) is optionally
    returned based on the `return_best_model` parameter.

    During training, the function prints and records training and validation losses
    for each epoch and plots a graph showing the loss evolution.

    Example:
    ```
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64)
    best_model = TrainLoopV2(model, optimizer, loss_function, 50, scheduler, train_dataloader, val_dataloader, 5)
    ```
    """
    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    total_train_loss = []
    total_val_loss = []
    best_model_weights = model.state_dict()

    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch : {epoch}\n----------------------")
        train_loss = 0
        for batch, (x1, x2, y) in enumerate(train_dataloader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            distance = model(x1, x2)
            loss = loss_function(distance, y)
            train_loss += loss
            loss.backward()
            optimizer.step()
            print(f"Loss for batch {batch} = {loss}")

        print("\nTraining Loss for epoch {} = {}\n".format(epoch, train_loss))
        total_train_loss.append(train_loss/len(train_dataloader.dataset))

        model.eval()
        validation_loss = 0
        with torch.inference_mode():
            for x1, x2, y in val_dataloader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)
                distance = model(x1,x2)
                loss = loss_function(distance, y)
                validation_loss+=loss

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_without_improvement = 0
                best_model_weights = model.state_dict()
            else:
                epochs_without_improvement+=1

            print(f"Current Validation Loss = {validation_loss}")
            print(f"Best Validation Loss = {best_val_loss}")
            print(f"Epochs without Improvement = {epochs_without_improvement}")

        total_val_loss.append(validation_loss/len(val_dataloader.dataset))
        if epochs_without_improvement >= early_stopping_rounds:
            print("Early Stoppping Triggered")
            break

        scheduler.step()

    if return_best_model == True:
        model.load_state_dict(best_model_weights)

    total_train_loss = [item.cpu().detach().numpy() for item in total_train_loss]
    total_val_loss = [item.cpu().detach().numpy() for item in total_val_loss]

    x = np.arange(len(total_train_loss))

    sns.set_style('whitegrid')
    plt.figure(figsize=(14,5))
    sns.lineplot(x=x, y=total_train_loss, label='Training Loss')
    sns.lineplot(x=x, y=total_val_loss, label='Validation Loss')
    plt.title("Loss over {} Epochs".format(len(total_train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(x)

    plt.show()
import os
import torch
import numpy as np
import pandas as pd
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
            y (torch.Tensor): The binary label indicating the similarity between the embeddings
                              (1 for similar pairs, 0 for dissimilar pairs).

        Returns:
            torch.Tensor: The contrastive loss value.

        Note:
            This function computes the contrastive loss for a pair of embeddings based on their distance and similarity
            label. The loss aims to pull similar pairs closer together and push dissimilar pairs apart in the embedding space.
        """
        return y*torch.pow(torch.tensor(dist), 2) + (1-y)*torch.pow(torch.max(torch.tensor([self.margin-dist, 0])), 2)
    
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
        image1_path = os.path.join(self.image_dir, str(self.labels.iat[index, 1]))
        image2_path = os.path.join(self.image_dir, str(self.labels.iat[index, 2]))
        label = torch.tensor(self.labels.iat[index,3])
        
        image1 = Image.open(image1_path).convert("L")
        image2 = Image.open(image2_path).convert("L")

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)        

        return image1, image2, label
    
    def __len__(self):
        return len(self.labels)
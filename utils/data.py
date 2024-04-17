import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

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
    def __init__(self, labels_csv:str, image_dir:str, transforms=None) -> None:
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
    
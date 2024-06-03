import os
import torch
import pandas as pd
from PIL import Image
from PIL.ImageOps import invert
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode

class SiameseDataset(Dataset):
    """
    Dataset for Siamese neural network training.

    Args:
        labels_csv (str): Path to the CSV file containing labels.
        image_dir (str): Path to the directory containing images.
        resize (int | list): Desired size for resizing images. If an int is provided, images
            will be resized to a square of this size. If a list is provided, it should contain
            two integers representing width and height respectively.
        train (bool, optional): Whether the dataset is for training or not. Defaults to True.

    Attributes:
        labels (DataFrame): DataFrame containing label information loaded from labels_csv.
        image_dir (str): Path to the directory containing images.
        transform (Compose): Composition of transformations to be applied to the images.

    Methods:
        __getitem__(self, index): Retrieves and preprocesses a pair of images along with their label.
        __len__(self): Returns the total number of samples in the dataset.
        get_transform(resize, train): Generates a composition of transformations based on the provided
            parameters.

    """
    def __init__(self, labels_csv: str, image_dir: str, resize: int | list, train: bool = True) -> None:
        super().__init__()
        self.labels = pd.read_csv(labels_csv, index_col=False)
        self.image_dir = image_dir
        self.transform = self.get_transform(resize, train)

    def __getitem__(self, index):
        image1_path = os.path.join(self.image_dir, str(self.labels.iat[index, 0]))
        image2_path = os.path.join(self.image_dir, str(self.labels.iat[index, 1]))
        label = torch.tensor(self.labels.iat[index,2])

        image1 = invert(Image.open(image1_path).convert("L"))
        image2 = invert(Image.open(image2_path).convert("L"))

        image1, image2 = self.transform(image1, image2)        

        return image1, image2, label
    
    def __len__(self):
        return len(self.labels)
    
    @staticmethod
    def get_transform(resize: int | list, train: bool = True):
        """
        Generates a composition of transformations to be applied to the images.

        Args:
            resize (int | list): Desired size for resizing images.
            train (bool, optional): Whether the transformation is for training or not. Defaults to True.

        Returns:
            Compose: Composition of transformations.

        """
        if type(resize) is list:
            resize = (resize[0], resize[1])
        else:
            resize = (resize, resize)
        transform = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
        ]
        if train:
            transform.extend([
                v2.RandomHorizontalFlip(0.55),
                v2.RandomVerticalFlip(0.55),
                v2.RandomRotation(45, InterpolationMode.BILINEAR)
            ])
        transform.append(v2.Normalize(mean=[0.5], std=[0.5]))

        return v2.Compose(transform)
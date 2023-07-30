import os
import torch
import pandas as pd
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split

class CustomDataset_CSVlabels(Dataset):
    """
    A PyTorch dataset for loading spectrogram images and their corresponding labels from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing the image file names and labels.
        img_dir (str): Root directory where the image files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, ``transforms.RandomCrop`` for randomly cropping an image.

    Attributes:
        img_labels (DataFrame): A pandas dataframe containing the image file names and labels.
        img_dir (str): Root directory where the image files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, ``transforms.RandomCrop`` for randomly cropping an image.
    
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the image and label at the given index.

    Returns:
        A PyTorch dataset object that can be passed to a DataLoader for batch processing.
    """
    def __init__(self,csv_file, img_dir, transform=None) -> None:
        super().__init__()
        self.img_labels = pd.read_csv(csv_file)
        self.img_labels.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.img_labels)
    
    def __getitem__(self, index):
        """
        Returns the image and label at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
        image = Image.open(img_path)
        image = image.convert("RGB")
        y_label = torch.tensor(int(self.img_labels.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
    
class CustomDataset_FolderLabels:
    """
    CustomDataset class for loading and splitting a dataset into training, validation, and testing sets.

    Args:
        data_path (str): Path to the main folder containing subfolders for each class.
        train_ratio (float): Ratio of data allocated for the training set (0.0 to 1.0).
        val_ratio (float): Ratio of data allocated for the validation set (0.0 to 1.0).
        test_ratio (float): Ratio of data allocated for the testing set (0.0 to 1.0).
        batch_size (int): Number of samples per batch in the data loaders.
        transform (torchvision.transforms.transforms.Compose): Transformations to be applied on the image

    Attributes:
        train_loader (torch.utils.data.DataLoader): Data loader for the training set.
        val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
        test_loader (torch.utils.data.DataLoader): Data loader for the testing set.

    """
    def __init__(self, data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32, transform=None):
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset and splits it into training, validation, and testing sets.

        """
        dataset = ImageFolder(root=self.data_path, transform=self.transform)
        num_samples = len(dataset)

        train_size = int(self.train_ratio * num_samples)
        val_size = int(self.val_ratio * num_samples)
        test_size = num_samples - train_size - val_size

        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_size, val_size, test_size])

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def get_train_loader(self):
        """
        Get the data loader for the training set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the training set.

        """
        return self.train_loader

    def get_val_loader(self):
        """
        Get the data loader for the validation set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the validation set.

        """
        return self.val_loader

    def get_test_loader(self):
        """
        Get the data loader for the testing set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the testing set.

        """
        return self.test_loader
    
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
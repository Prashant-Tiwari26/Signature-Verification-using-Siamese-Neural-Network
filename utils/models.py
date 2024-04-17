import torch

class Model_LRN(torch.nn.Module):
    """
    Siamese CNN model with Local Response Normalization.

    This model consists of three shared convolutional layers with Local Response Normalization
    followed by Fully Connected layers for feature extraction. The architecture is designed
    to take two input images (x_genuine and x_test) and process them through the same
    set of convolutional layers before extracting feature vectors.

    Args:
        None

    Returns:
        None

    Shape:
        - Input:
            - x_genuine (torch.Tensor): A 4D tensor representing a batch of genuine images
              with shape (batch_size, channels, height, width).
            - x_test (torch.Tensor): A 4D tensor representing a batch of test images
              with shape (batch_size, channels, height, width).

        - Output:
            - y_genuine (torch.Tensor): A 2D tensor representing the feature vectors of the
              genuine images with shape (batch_size, 128).
            - y_test (torch.Tensor): A 2D tensor representing the feature vectors of the
              test images with shape (batch_size, 128).

    Model Architecture:
        --------------------------------------------------------------------
        Layer (type)            Output Shape                        Param #
        ====================================================================
        Conv2d-1                [batch_size, 96, 210, 145]          34,944
        ReLU-2                  [batch_size, 96, 210, 145]               0
        LocalResponseNorm-3     [batch_size, 96, 210, 145]             192
        ReLU-4                  [batch_size, 96, 210, 145]               0
        MaxPool2d-5             [batch_size, 96, 104, 72]                0
        Conv2d-6                [batch_size, 256, 104, 72]         614,656
        ReLU-7                  [batch_size, 256, 104, 72]               0
        LocalResponseNorm-8     [batch_size, 256, 104, 72]             512
        ReLU-9                  [batch_size, 256, 104, 72]               0
        MaxPool2d-10            [batch_size, 256, 51, 35]                0
        Dropout2d-11            [batch_size, 256, 51, 35]                0
        Conv2d-12               [batch_size, 384, 51, 35]          885,120
        ReLU-13                 [batch_size, 384, 51, 35]                0
        Conv2d-14               [batch_size, 256, 51, 35]          884,992
        ReLU-15                 [batch_size, 256, 51, 35]                0
        MaxPool2d-16            [batch_size, 256, 25, 17]                0
        Dropout2d-17            [batch_size, 256, 25, 17]                0
        Flatten-18              [batch_size, 108800]                     0
        Linear-19               [batch_size, 1024]             111,412,224
        ReLU-20                 [batch_size, 1024]                       0
        Dropout1d-21            [batch_size, 1024]                       0
        Linear-22               [batch_size, 128]                  131,200
        ===================================================================
        Total params: 113,963,840
        Trainable params: 113,963,840
        Non-trainable params: 0
        -------------------------------------------------------------------

    Note:
        ReLU activation is used after each convolutional layer, and Dropout is applied
        for regularization to prevent overfitting.

    Example:
        # Create an instance of the Model_LRN
        model = Model_LRN()

        # Assuming you have loaded the genuine and test images as tensors
        genuine_images = ...  # Tensor of genuine images
        test_images = ...  # Tensor of test images

        # Pass the images through the model
        feature_vectors_genuine, feature_vectors_test = model(genuine_images, test_images)
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_branch = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=1),
            torch.nn.ReLU(),
            torch.nn.LocalResponseNorm(alpha=0.0001, beta=0.75, size=5, k=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.LocalResponseNorm(alpha=0.0001, beta=0.75, size=5, k=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Dropout2d(p=0.3),
            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Dropout2d(p=0.3),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=108800, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Dropout1d(p=0.5),
            torch.nn.Linear(in_features=1024, out_features=128)
        )

    def forward(self, x_genuine, x_test):
        """
        Forward pass of the Siamese CNN model.

        Args:
            x_genuine (torch.Tensor): A batch of genuine images with shape (batch_size, channels, height, width).
            x_test (torch.Tensor): A batch of test images with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A tensor representing the feature vectors of the genuine images with shape (batch_size, 128).
            torch.Tensor: A tensor representing the feature vectors of the test images with shape (batch_size, 128).
        """
        y_genuine = self.model_branch(x_genuine)
        y_test = self.model_branch(x_test)
        return torch.nn.functional.pairwise_distance(y_genuine, y_test)

class Model_BN(torch.nn.Module):
    """
    Siamese CNN model with Batch Normalization.

    This model consists of three shared convolutional layers with Batch Normalization
    followed by Fully Connected layers for feature extraction. The architecture is designed
    to take two input images (x_genuine and x_test) and process them through the same
    set of convolutional layers before extracting feature vectors.

    Args:
        None

    Returns:
        None

    Shape:
        - Input:
            - x_genuine (torch.Tensor): A 4D tensor representing a batch of genuine images
              with shape (batch_size, channels, height, width).
            - x_test (torch.Tensor): A 4D tensor representing a batch of test images
              with shape (batch_size, channels, height, width).

        - Output:
            - y_genuine (torch.Tensor): A 2D tensor representing the feature vectors of the
              genuine images with shape (batch_size, 128).
            - y_test (torch.Tensor): A 2D tensor representing the feature vectors of the
              test images with shape (batch_size, 128).

    Model Architecture:
        --------------------------------------------------------------------
        Layer (type)            Output Shape                        Param #
        ====================================================================
        Conv2d-1                [batch_size, 96, 210, 145]          34,944
        SELU-2                  [batch_size, 96, 210, 145]               0
        BatchNorm2d-3           [batch_size, 96, 210, 145]             192
        SELU-4                  [batch_size, 96, 210, 145]               0
        MaxPool2d-5             [batch_size, 96, 104, 72]                0
        Conv2d-6                [batch_size, 256, 104, 72]         614,656
        SELU-7                  [batch_size, 256, 104, 72]               0
        BatchNorm2d-8           [batch_size, 256, 104, 72]             512
        SELU-9                  [batch_size, 256, 104, 72]               0
        MaxPool2d-10            [batch_size, 256, 51, 35]                0
        Dropout2d-11            [batch_size, 256, 51, 35]                0
        Conv2d-12               [batch_size, 384, 51, 35]          885,120
        SELU-13                 [batch_size, 384, 51, 35]                0
        Conv2d-14               [batch_size, 256, 51, 35]          884,992
        SELU-15                 [batch_size, 256, 51, 35]                0
        MaxPool2d-16            [batch_size, 256, 25, 17]                0
        Dropout2d-17            [batch_size, 256, 25, 17]                0
        Flatten-18              [batch_size, 108800]                     0
        Linear-19               [batch_size, 1024]             111,412,224
        SELU-20                 [batch_size, 1024]                       0
        Dropout1d-21            [batch_size, 1024]                       0
        Linear-22               [batch_size, 128]                  131,200
        ===================================================================
        Total params: 113,963,840
        Trainable params: 113,963,840
        Non-trainable params: 0
        -------------------------------------------------------------------

    Note:
        SELU activation is used after each convolutional layer, and Dropout is applied
        for regularization to prevent overfitting.

    Example:
        # Create an instance of the Model_BN
        model = Model_BN()

        # Assuming you have loaded the genuine and test images as tensors
        genuine_images = ...  # Tensor of genuine images
        test_images = ...  # Tensor of test images

        # Pass the images through the model
        feature_vectors_genuine, feature_vectors_test = model(genuine_images, test_images)
    """
    def __init__(self) -> None:
        super().__init__()
        self.model_branch = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=1),
            torch.nn.SELU(),
            torch.nn.BatchNorm2d(num_features=96),
            torch.nn.SELU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            torch.nn.SELU(),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.SELU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Dropout2d(p=0.3),
            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            torch.nn.SELU(),
            torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.SELU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Dropout2d(p=0.3),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=108800, out_features=1024),
            torch.nn.SELU(),
            torch.nn.Dropout1d(p=0.5),
            torch.nn.Linear(in_features=1024, out_features=128)
        )

    def forward(self, x_genuine, x_test):
        """
        Forward pass of the Siamese CNN model.

        Args:
            x_genuine (torch.Tensor): A batch of genuine images with shape (batch_size, channels, height, width).
            x_test (torch.Tensor): A batch of test images with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A tensor representing the feature vectors of the genuine images with shape (batch_size, 128).
            torch.Tensor: A tensor representing the feature vectors of the test images with shape (batch_size, 128).
        """
        y_genuine = self.model_branch(x_genuine)
        y_test = self.model_branch(x_test)
        return torch.nn.functional.pairwise_distance(y_genuine, y_test)

class Model_BN_s(torch.nn.Module):
    """
    Siamese CNN model with Batch Normalization.

    This model consists of three shared convolutional layers with Batch Normalization
    followed by Fully Connected layers for feature extraction. The architecture is designed
    to take two input images (x_genuine and x_test) and process them through the same
    set of convolutional layers before extracting feature vectors.

    Args:
        None

    Returns:
        None

    Shape:
        - Input:
            - x_genuine (torch.Tensor): A 4D tensor representing a batch of genuine images
              with shape (batch_size, channels, height, width).
            - x_test (torch.Tensor): A 4D tensor representing a batch of test images
              with shape (batch_size, channels, height, width).

        - Output:
            - y_genuine (torch.Tensor): A 2D tensor representing the feature vectors of the
              genuine images with shape (batch_size, 128).
            - y_test (torch.Tensor): A 2D tensor representing the feature vectors of the
              test images with shape (batch_size, 128).

    Model Architecture:
        --------------------------------------------------------------------
        Layer (type)            Output Shape                        Param #
        ====================================================================
        Conv2d-1                [batch_size, 96, 210, 145]          34,944
        SELU-2                  [batch_size, 96, 210, 145]               0
        BatchNorm2d-3           [batch_size, 96, 210, 145]             192
        SELU-4                  [batch_size, 96, 210, 145]               0
        MaxPool2d-5             [batch_size, 96, 104, 72]                0
        Conv2d-6                [batch_size, 256, 104, 72]         614,656
        SELU-7                  [batch_size, 256, 104, 72]               0
        BatchNorm2d-8           [batch_size, 256, 104, 72]             512
        SELU-9                  [batch_size, 256, 104, 72]               0
        MaxPool2d-10            [batch_size, 256, 51, 35]                0
        Dropout2d-11            [batch_size, 256, 51, 35]                0
        Conv2d-12               [batch_size, 384, 51, 35]          885,120
        SELU-13                 [batch_size, 384, 51, 35]                0
        Conv2d-14               [batch_size, 256, 51, 35]          884,992
        SELU-15                 [batch_size, 256, 51, 35]                0
        MaxPool2d-16            [batch_size, 256, 25, 17]                0
        Conv2d-17               [batch_size, 256, 17, 25]          590,080
        SELU-18                 [batch_size, 256, 17, 25]                0
        MaxPool2d-19            [batch_size, 256, 8, 12]                 0
        Dropout2d-20            [batch_size, 256, 8, 12]                 0
        Flatten-21              [batch_size, 24576]                      0
        Linear-22               [batch_size, 1024]              25,166,848
        SELU-23                 [batch_size, 1024]                       0
        Dropout1d-24            [batch_size, 1024]                       0
        Linear-25               [batch_size, 128]                  131,200
        ===================================================================
        Total params: 28,285,312
        Trainable params: 28,285,312
        Non-trainable params: 0
        -------------------------------------------------------------------

    Note:
        SELU activation is used after each convolutional layer, and Dropout is applied
        for regularization to prevent overfitting.

    Example:
        # Create an instance of the Model_BN
        model = Model_BN()

        # Assuming you have loaded the genuine and test images as tensors
        genuine_images = ...  # Tensor of genuine images
        test_images = ...  # Tensor of test images

        # Pass the images through the model
        feature_vectors_genuine, feature_vectors_test = model(genuine_images, test_images)
    """
    def __init__(self) -> None:
        super().__init__()
        self.model_branch = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=1),
            torch.nn.SELU(),
            torch.nn.BatchNorm2d(num_features=96),
            torch.nn.SELU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            torch.nn.SELU(),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.SELU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Dropout2d(p=0.3),
            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            torch.nn.SELU(),
            torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.SELU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.SELU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Dropout2d(p=0.3),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=24576, out_features=1024),
            torch.nn.SELU(),
            torch.nn.Dropout1d(p=0.5),
            torch.nn.Linear(in_features=1024, out_features=128)
        )

    def forward(self, x_genuine, x_test):
        """
        Forward pass of the Siamese CNN model.

        Args:
            x_genuine (torch.Tensor): A batch of genuine images with shape (batch_size, channels, height, width).
            x_test (torch.Tensor): A batch of test images with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: A tensor representing the feature vectors of the genuine images with shape (batch_size, 128).
            torch.Tensor: A tensor representing the feature vectors of the test images with shape (batch_size, 128).
        """
        y_genuine = self.model_branch(x_genuine)
        y_test = self.model_branch(x_test)
        return torch.nn.functional.pairwise_distance(y_genuine, y_test)
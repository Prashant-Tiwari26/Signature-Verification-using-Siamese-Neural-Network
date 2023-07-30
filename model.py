import torch

class Model_LRN(torch.nn.Module):
    """
    Custom Siamese CNN model with Local Response Normalization (LRN).

    This model consists of two branches, one for processing genuine signatures and the other
    for processing test (query) signatures. Both branches share the same architecture.

    Parameters:
        None

    Attributes:
        genuine_arm (torch.nn.Sequential): Sequential container representing the architecture for processing genuine signatures.
        test_arm (torch.nn.Sequential): Sequential container representing the architecture for processing test (query) signatures.

    Methods:
        forward(x_genuine, x_test): Forward pass through the model.

    Example:
        # Create an instance of the Model_LRN class
        model = Model_LRN()

        # Forward pass with input data (genuine and test signatures)
        genuine_data = torch.randn(64, 3, 224, 224)
        test_data = torch.randn(64, 3, 224, 224)
        genuine_output, test_output = model(genuine_data, test_data)
    """

    def __init__(self) -> None:
        super().__init__()
        self.genuine_arm = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=1),
            torch.nn.SELU(),
            torch.nn.LocalResponseNorm(alpha=0.0001, beta=0.75, size=5, k=2),
            torch.nn.SELU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            torch.nn.SELU(),
            torch.nn.LocalResponseNorm(alpha=0.0001, beta=0.75, size=5, k=2),
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
            torch.nn.Dropout1d(p=0.5),
            torch.nn.Linear(in_features=1024, out_features=128)
        )

        self.test_arm = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=1),
            torch.nn.SELU(),
            torch.nn.LocalResponseNorm(alpha=0.0001, beta=0.75, size=5, k=2),
            torch.nn.SELU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            torch.nn.SELU(),
            torch.nn.LocalResponseNorm(alpha=0.0001, beta=0.75, size=5, k=2),
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
            torch.nn.Dropout1d(p=0.5),
            torch.nn.Linear(in_features=1024, out_features=128)
        )

    def forward(self, x_genuine, x_test):
        """
        Forward pass through the model.

        Parameters:
            x_genuine (torch.Tensor): Input data representing genuine signatures.
            x_test (torch.Tensor): Input data representing test (query) signatures.

        Returns:
            tuple: A tuple containing two torch.Tensors representing the output of processing
                   genuine and test signatures, respectively.
        """

        y_genuine = self.genuine_arm(x_genuine)
        y_test = self.test_arm(x_test)
        return y_genuine, y_test

class Model_BN(torch.nn.Module):
    """
    Custom Siamese CNN model with Batch Normalization(BN).

    This model consists of two branches, one for processing genuine signatures and the other
    for processing test (query) signatures. Both branches share the same architecture.

    Parameters:
        None

    Attributes:
        genuine_arm (torch.nn.Sequential): Sequential container representing the architecture for processing genuine signatures.
        test_arm (torch.nn.Sequential): Sequential container representing the architecture for processing test (query) signatures.

    Methods:
        forward(x_genuine, x_test): Forward pass through the model.

    Example:
        # Create an instance of the Model_BN class
        model = Model_BN()

        # Forward pass with input data (genuine and test signatures)
        genuine_data = torch.randn(64, 3, 224, 224)
        test_data = torch.randn(64, 3, 224, 224)
        genuine_output, test_output = model(genuine_data, test_data)
    """

    def __init__(self) -> None:
        super().__init__()
        self.genuine_arm = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=1),
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
            torch.nn.Dropout1d(p=0.5),
            torch.nn.Linear(in_features=1024, out_features=128)
        )

        self.test_arm = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=1),
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
            torch.nn.Dropout1d(p=0.5),
            torch.nn.Linear(in_features=1024, out_features=128)
        )

    def forward(self, x_genuine, x_test):
        """
        Forward pass through the model.

        Parameters:
            x_genuine (torch.Tensor): Input data representing genuine signatures.
            x_test (torch.Tensor): Input data representing test (query) signatures.

        Returns:
            tuple: A tuple containing two torch.Tensors representing the output of processing
                   genuine and test signatures, respectively.
        """

        y_genuine = self.genuine_arm(x_genuine)
        y_test = self.test_arm(x_test)
        return y_genuine, y_test

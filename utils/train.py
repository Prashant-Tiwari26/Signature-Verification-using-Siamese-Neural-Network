import torch
import logging
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from timeit import default_timer as timer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

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

def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    scheduler,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    fig_save_path: str,
    num_epochs: int = 20,
    batch_loss: int = 1,
    early_stopping_rounds: int = 5,
    return_best_model: bool = True,
    device: str = 'cuda'
):
    """
    Function to train a siamese neural network model using a specified loss function, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The siamese neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model parameters.
        loss_function (torch.nn.Module): The loss function used for computing the loss.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        scheduler: The learning rate scheduler.
        fig_save_path (str): Path to save the training progress plot.
        num_epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_loss (int, optional): Number of batches to accumulate loss before backpropagation. Defaults to 1.
        early_stopping_rounds (int, optional): Number of epochs without improvement before early stopping. Defaults to 5.
        return_best_model (bool, optional): Whether to return the model with the best validation loss. Defaults to True.
        device (str, optional): Device to run the training on (e.g., 'cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        torch.nn.Module or None: Trained model if `return_best_model` is True, otherwise None.

    """
    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    total_train_loss = []
    total_val_loss = []
    val_accuracies = []

    best_model_weights = model.state_dict()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        print("\n---------------------\n")
        logger.info("Epoch {} | Learning Rate = {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_loss = 0
        for i, (x1, x2, y) in enumerate(train_dataloader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            distance = model(x1, x2)
            loss = loss_function(distance, y)
            train_loss += loss
            loss.backward()
            optimizer.step()
            if i % batch_loss == 0:
                logger.info("Loss for batch {} = {}".format(i, loss))

        logger.info("\nTraining Loss for epoch {} = {}\n".format(epoch, train_loss))
        total_train_loss.append(train_loss/len(train_dataloader.dataset))

        model.eval()
        validation_loss = 0
        with torch.inference_mode():
            for x1, x2, y in val_dataloader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                distance = model(x1,x2)
                loss = loss_function(distance, y)
                validation_loss+=loss

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                epochs_without_improvement = 0
                best_model_weights = model.state_dict()
            else:
                epochs_without_improvement+=1

            logger.info(f"Current Validation Loss = {validation_loss}")
            logger.info(f"Best Validation Loss = {best_val_loss}")
            logger.info(f"Epochs without Improvement = {epochs_without_improvement}")

        total_val_loss.append(validation_loss/len(val_dataloader.dataset))
        
        if epochs_without_improvement >= early_stopping_rounds:
            logger.warning("Early Stoppping Triggered")
            break

        try:
            scheduler.step()
        except:
            scheduler.step(validation_loss)

    if return_best_model == True:
        model.load_state_dict(best_model_weights)

    total_train_loss = [item.cpu().detach().numpy() for item in total_train_loss]
    total_val_loss = [item.cpu().detach().numpy() for item in total_val_loss]

    total_train_loss = np.array(total_train_loss)
    total_val_loss = np.array(total_val_loss)
    val_accuracies = np.array(val_accuracies)

    # fig, ax1 = plt.subplots(figsize=(10, 7.5))
    plt.figure(figsize=(10,7.5), dpi=150)
    sns.lineplot(x=range(len(total_train_loss)), y=total_train_loss, label='Training Loss')
    sns.lineplot(x=range(len(total_val_loss)), y=total_val_loss, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.title("Loss and accuracy during training")
    plt.subplots_adjust(wspace=0.3)
    plt.grid(True, linestyle='--')
    plt.xticks(range(len(total_train_loss)), rotation=45)
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    plt.show()
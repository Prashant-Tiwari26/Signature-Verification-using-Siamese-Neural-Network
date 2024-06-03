import os
import sys
import torch
import logging
import numpy as np
import seaborn as sns
from tqdm import tqdm
from typing import Literal
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from src.models.cnn import MODELS
from src.data import SiameseDataset
from src.utils.loss import ContrastiveLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def lr_lambda(epoch):
    if epoch <= 15:
        return 0.85
    return 1

@dataclass
class TrainingConfig:
    model: Literal['shufflenet', 'custom', 'custom_large']
    img_dir: str
    resize: int
    train_csv_path: str
    val_csv_path: str
    num_epochs: int
    batch_size: int
    batch_loss: int
    early_stopping_rounds: int
    model_save_dir: str
    training_loss_graph_path: str
    return_best_model: bool = True
    device: str = 'cuda'
    task_type: str = 'train'
    learning_rate: float = 1e-4

class ModelTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.model = MODELS[self.config.model]
        self.model.to(self.config.device)
        logger.info("{} Model loaded".format(config.model))
        logger.info("Number of Parameters = {:,}".format(self._get_num_params()))
        self.__get_dataloaders()
        self.loss_fn = ContrastiveLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), float(config.learning_rate))
        self.lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda)
        logger.info('Loss function, optimizer and learning rate scheduler prepared\n\n')

    def _get_num_params(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def __get_dataloaders(self):
        train_data = SiameseDataset(self.config.train_csv_path, self.config.img_dir, self.config.resize)
        val_data = SiameseDataset(self.config.val_csv_path, self.config.img_dir, self.config.resize, False)
        self.train_dataloader = DataLoader(train_data, self.config.batch_size, True)
        self.val_dataloader = DataLoader(val_data, self.config.batch_size, True)
        logger.info("No. of batches in training set = {}".format(len(self.train_dataloader)))
        logger.info("No. of batches in validation set = {}".format(len(self.val_dataloader)))
        logger.info('DataLoaders prepared')

    def train(self):
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        total_train_loss = []
        total_val_loss = []

        best_model_weights = self.model.state_dict()

        for epoch in tqdm(range(self.config.num_epochs)):
            self.model.train()
            print("---------------------")
            logger.info("Epoch {} | Learning Rate = {}".format(epoch, self.optimizer.param_groups[0]['lr']))
            train_loss = 0
            for i, (x1, x2, y) in enumerate(self.train_dataloader):
                x1, x2, y = x1.to(self.config.device), x2.to(self.config.device), y.to(self.config.device)
                self.optimizer.zero_grad()
                distance = self.model(x1, x2)
                loss = self.loss_fn(distance, y)
                train_loss += loss
                loss.backward()
                self.optimizer.step()
                if i % self.config.batch_loss == 0 or i == len(self.train_dataloader) - 1:
                    logger.info("Loss for batch {} = {}".format(i, loss))

            logger.info("Training Loss for epoch {} = {}\n".format(epoch, train_loss))
            total_train_loss.append(train_loss/len(self.train_dataloader.dataset))

            self.model.eval()
            validation_loss = 0
            with torch.inference_mode():
                for x1, x2, y in self.val_dataloader:
                    x1, x2, y = x1.to(self.config.device), x2.to(self.config.device), y.to(self.config.device)
                    distance = self.model(x1,x2)
                    loss = self.loss_fn(distance, y)
                    validation_loss+=loss

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    epochs_without_improvement = 0
                    best_model_weights = self.model.state_dict()
                    logger.info('Best model weights updated')
                else:
                    epochs_without_improvement+=1

                logger.info(f"Current Validation Loss = {validation_loss}")
                logger.info(f"Best Validation Loss = {best_val_loss}")
                logger.info(f"Epochs without Improvement = {epochs_without_improvement}")

            total_val_loss.append(validation_loss/len(self.val_dataloader.dataset))
            
            if epochs_without_improvement >= self.config.early_stopping_rounds:
                logger.warning("Early Stoppping Triggered")
                break

            try:
                self.lr_scheduler.step()
            except:
                self.lr_scheduler.step(validation_loss)
        logger.info('Training finished')
        
        best_path = self.config.model_save_dir + '/' + self.config.model + '_best.pth'
        torch.save(best_model_weights, best_path)
        logger.info('Best model weights saved to {}'.format(best_path))

        if self.config.return_best_model == True:
            self.model.load_state_dict(best_model_weights)

        total_train_loss = [item.cpu().detach().numpy() for item in total_train_loss]
        total_val_loss = [item.cpu().detach().numpy() for item in total_val_loss]

        total_train_loss = np.array(total_train_loss)
        total_val_loss = np.array(total_val_loss)

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
        plt.savefig(self.config.training_loss_graph_path, dpi=300, bbox_inches='tight')
        logger.info("Training and validation loss graph saved at {}".format(self.config.training_loss_graph_path))
        plt.show()


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
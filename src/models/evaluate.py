import os
import sys
import json
import torch
import logging
import numpy as np
import pandas as pd
from typing import Literal
from sklearn import metrics
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

@dataclass
class EvaluationConfig:
    model_name: Literal['shufflenet', 'custom_large', 'custom']
    model_weights_path: str
    eval_csv_path: str
    img_dir: str
    distance_threshold: float
    batch_size: int
    resize: int
    cf_save_path: str
    report_save_path: str
    device: str = 'cuda'
    task_type: str = 'eval'

class ModelEvaluation:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

        self.model = MODELS[config.model_name]
        self.model.load_state_dict(torch.load(config.model_weights_path))
        self.model.to(config.device)
        self.model.eval()
        logger.info("Model prepared for evaluation")
        
        self.__get_dataloaders()

    def __get_dataloaders(self):
        eval_data = SiameseDataset(self.config.eval_csv_path, self.config.img_dir, self.config.resize, False)
        self.eval_dataloader = DataLoader(eval_data, self.config.batch_size, True)
        logger.info("No. of batches in evaluation set = {}".format(len(self.eval_dataloader)))
        logger.info("DataLoader prepared")

    def evaluate(self):
        self._get_distances()
        self._get_preds()
        self._get_metrics()
        logger.info("Model evaluation finished")

    def _get_distances(self):
        self.distances = []
        self.labels = []
        with torch.inference_mode():
            for x1, x2, y in self.eval_dataloader:
                x1 = x1.to(self.config.device)
                x2 = x2.to(self.config.device)
                outputs = self.model(x1, x2)
                self.distances.extend(outputs.cpu().numpy())
                self.labels.extend(y.cpu().numpy())
        logger.info("Distances calculated")
        self.distances = np.array(self.distances)
        self.labels = np.array(self.labels)
    
    def _get_preds(self):
        self.preds = self.distances.copy()
        for i in range(len(self.preds)):
            if self.preds[i] > self.config.distance_threshold:
                self.preds[i] = 0
            else:
                self.preds[i] = 1
        logger.info("Predictions calculated")

    def _get_metrics(self):
        report = metrics.classification_report(self.labels, self.preds, output_dict=True)
        logger.info("Classification report:\n{}".format(report))
        with open(self.config.report_save_path, 'w') as file:
            json.dump(report, file, indent=4)
            logger.info("Saved classification report at {}".format(self.config.report_save_path))
        
        a = metrics.ConfusionMatrixDisplay.from_predictions(self.labels, self.preds, cmap='Reds')
        plt.title("{} confusion Matrix".format(self.config.model_name))
        plt.savefig(self.config.cf_save_path, dpi=300, bbox_inches = 'tight')
        logger.info("Saved confusion matrix at {}".format(self.config.cf_save_path))
        plt.show()
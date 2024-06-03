import sys
import yaml
import logging
from src.models.train import ModelTrainer, TrainingConfig
from src.models.evaluate import EvaluationConfig, ModelEvaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def load_config(config_path:str):
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data

def run_operations(trainconfig: TrainingConfig = None, evalconfig: EvaluationConfig = None):
    if trainconfig is not None:
        trainer = ModelTrainer(trainconfig)
        trainer.train()
    if evalconfig is not None:
        evaluate = ModelEvaluation(evalconfig)
        evaluate.evaluate()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        logger.info("Usage: python __main__.py train.yaml or python __main__.py eval.yaml or python __main__.py train.yaml eval.yaml")
        logger.error("Config file path not provided")
        sys.exit(1)
    elif len(sys.argv) == 2:
        config_data = load_config(sys.argv[1])
        if config_data['task_type'] == 'train':
            config = TrainingConfig(**config_data)
            run_operations(trainconfig=config)
        elif config_data['task_type'] == 'eval':
            config = EvaluationConfig(**config_data)
            run_operations(evalconfig=config)
    else:
        config_1 = load_config(sys.argv[1])
        config_2 = load_config(sys.argv[2])
        if config_1['task_type'] == 'train':
            run_operations(TrainingConfig(**config_1), EvaluationConfig(**config_2))
        else:
            run_operations(TrainingConfig(**config_2), EvaluationConfig(**config_1))
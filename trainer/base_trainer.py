from abc import ABC, abstractmethod
from configs.input_config import InputConfig
from repo.mlflow_repo import MLFRepo
from reader.factory import ReaderFactory
from loguru import logger
import torch
from db import get_collections


class BaseTrainer(ABC):
    def __init__(self, config: InputConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_data = get_collections()
        self.proj_id = config.proj_id
        self.username = config.username
        self.exp_id = config.exp_id
        self.data_config = config.data
        self.learning_rate = config.model.learning_rate
        self.num_epochs = config.model.num_epochs
        self.type = config.model.type
        self.train_batch_size = config.model.train_batch_size
        self.test_batch_size = config.model.test_batch_size
        self.num_epochs = config.model.num_epochs
        self.learning_rate = config.model.learning_rate
        self.threshold = config.model.threshold
        self.type = config.model.type
        self.model_name = f"{self.proj_id.lower()}"
        self.repo = MLFRepo(self.exp_id)
        # Check and tag all the untagged/unsuccessful runs from previous tries
        try:
            self.repo.check_and_tag()
        except Exception as e:
            logger.info(
                f"Failed to tag runs in {self.exp_id}. Ignore if this is a fresh exp \n {e}"
            )
        self.reader = ReaderFactory.create(self.data_config, self.exp_id, self.type)
        self.trainloader = self.reader.read(self.train_batch_size, False)
        self.testloader = self.reader.read(self.test_batch_size, True)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate_fn(self):
        pass

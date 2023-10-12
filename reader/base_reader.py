import os
import zipfile
from pathlib import Path
from loguru import logger
from abc import ABC, abstractmethod
from configs.data_config import DataConfig
from storage.factory import StorageFactory
import pandas as pd
from sklearn.model_selection import train_test_split


class DataReader(ABC):
    """Data Reader class to support multiple formats"""

    def __init__(self, config: DataConfig, exp_id: str) -> None:
        self.exp_id = exp_id
        self.config = config
        self.input = self.config.input
        self.target = self.config.target
        self.file_path = f"temp/{self.exp_id}/data.csv"
        self.folder_path = os.path.dirname(self.file_path)
        self.storage = StorageFactory.create(self.file_path, self.input)
        self.storage.download()
        self.train_ratio = self.config.test_split.train_ratio
        self.test_ratio = self.config.test_split.test_ratio
        self.preprocess()

    def preprocess(self) -> None:
        try:
            self.df = pd.read_csv(self.file_path)
            self.df["len"] = self.df["text"].apply(lambda x: len(x.split(" ")))
            self.df = self.df[self.df["len"] <= self.df["len"].quantile(0.9)]
            self.df.drop(columns=["len"], inplace=True)

        except Exception as e:
            logger.info(f"Error: {e}")

        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_ratio, stratify=y, random_state=29
        )

        self.train_df = pd.concat([X_train, y_train], axis=1)
        self.test_df = pd.concat([X_test, y_test], axis=1)

    @abstractmethod
    def read(self, eval_flag=False):
        pass

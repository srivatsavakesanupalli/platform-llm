import os
import mlflow
from loguru import logger
from abc import ABC, abstractmethod
import numpy as np
import torch
from PIL import Image
from db import get_collections


class BaseInfer(ABC):
    def __init__(self) -> None:
        self.model_name = os.environ['MODEL_NAME']
        self.vendor = os.getenv('VENDOR', 'aws')
        self.batch_input = os.getenv('BATCH_INPUT')
        self.model_version = os.environ['STATE']
        self.exp_id = os.environ['EXP_ID']
        self.batch_size = 2
        logger.info("Model loaded!!!")
        self.db = get_collections()['db']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_run_id(self):
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(self.model_name)
        run_id = None
        for version in versions:
            ver_dict = dict(version)
            if ver_dict['current_stage'] == self.model_version:
                run_id = ver_dict['run_id']
        return run_id
    
    def get_current_step(self, run_id, metric_key):
        client = mlflow.MlflowClient()
        metrics = client.get_metric_history(run_id=run_id, key=metric_key)
        try:
            step = max([metric.step for metric in metrics])+1
        except:
            step = 0
        return step


    def download_folder(self, artifacts_folder):
        run_id = self.get_run_id()
        logger.info(run_id)
        if run_id:
            logger.info(
                f'Run ID for the model name {self.model_name}:{self.model_version} is {run_id}')
            try:
                _ = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/{artifacts_folder}", dst_path='temp/mlflow')
                logger.info('Succefully downloaded model files')
            except Exception as e:
                logger.info(f'Download failed : {e}')
        else:
            logger.info(f'Run ID for the model name {self.model_name}:{self.model_version} not found')
    
    def get_train_embs(self):
        run_id = self.get_run_id()
        if run_id:
            logger.info(
                f'Run ID for the model name {self.model_name}:{self.model_version} is {run_id}')
            try:
                _ = mlflow.artifacts.download_artifacts(
                    f"runs:/{run_id}/embeddings", dst_path='temp/mlflow')
                with open('temp/mlflow/embeddings/train_embeddings.npy', 'rb') as f:
                    train_emb = np.load(f)
                logger.info('Succefully loaded the explainer')
            except Exception as e:
                logger.info(f'Explainer loading failed : {e}')
                train_emb = None
        else:
            train_emb = None
            logger.info(
                f'Run ID for the model name {self.model_name}:{self.model_version} not found')
        return train_emb

    def log_predictions(self):
        run_id = self.get_run_id()
        if run_id:
            logger.info(
                f'Run ID for the model name {self.model_name}: {self.model_version} is {run_id}')
            try:
                mlflow.end_run()  # end any existing runs
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifacts('predictions', 'predictions')
                return True
            except Exception as e:
                logger.info(f'Logging predictions failed : {e}')
        else:
            logger.info(
                f'Run ID for the model name {self.model_name}:{self.model_version} not found')

    @abstractmethod
    def infer(self, image: Image):
        pass

    @abstractmethod
    def explain(self, image: Image):
        pass

    @abstractmethod
    def infer_batch(self):
        pass

    @abstractmethod
    def preprocess(self, image: Image):
        pass

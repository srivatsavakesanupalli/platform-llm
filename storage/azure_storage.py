from .base_storage import BaseStorage
from loguru import logger
import os
import urllib.request


class AZUREStorage(BaseStorage):
    def __init__(self, filepath: str, object_path: str) -> None:
        super().__init__(filepath, object_path)

    def upload(self):
        return 'NotImplemented'

    def download(self):
        '''Currently only supports SAS urls or publicly accessible urls'''
        logger.info(f'Downloading data to {self.filepath}')
        experiment_folder = os.path.dirname(self.filepath)
        os.makedirs(experiment_folder, exist_ok=True)
        urllib.request.urlretrieve(
            self.object_path, filename=self.filepath)

    def is_modified(self) -> bool:
        logger.info('Checking for file changes')
        return True

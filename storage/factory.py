import importlib
import os
from urllib.parse import urlparse
from loguru import logger

STORAGE_BACKEND = os.getenv('STORAGE_BACKEND', 'azure')


class StorageFactory:
    @staticmethod
    def create(filepath: str, object_key: str):
        try:
            logger.info(object_key)
            backend = StorageFactory.get_storage_backend(object_key)
            logger.info(backend)
            logger.info("---> in Storage Factory")
            class_name = f'{backend.upper()}Storage'
            module = importlib.import_module(
                f'storage.{backend.lower()}_storage')
            class_ = getattr(module, class_name)
            instance = class_(filepath, object_key)
            return instance
        except (ImportError, AttributeError):
            raise NotImplementedError(
                f'Storage Backend {backend} is not supported')
    
    @staticmethod
    def get_storage_backend(url: str):
        parsed_url = urlparse(url)
        scheme = parsed_url.scheme
        if scheme == "s3":
            return "AWS"
        elif scheme in ["https", "http"]:
            return "Azure"
        else:
            return "Azure"

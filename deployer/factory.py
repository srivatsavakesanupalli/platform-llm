import importlib
from configs.deployer_config import DeployerConfig
import os

DEPLOYER_BACKEND = os.getenv('DEPLOYER_BACKEND', 'aws')


class DeployFactory:
    @staticmethod
    def create(config: DeployerConfig, backend: str = DEPLOYER_BACKEND):
        try:
            class_name = f'{backend.upper()}Deployer'
            module = importlib.import_module(
                f'deployer.{backend}_deployer')
            class_ = getattr(module, class_name)
            instance = class_(config)
            return instance
        except (ImportError, AttributeError):
            raise NotImplementedError(
                f'Deployer Backend {backend} is not supported')

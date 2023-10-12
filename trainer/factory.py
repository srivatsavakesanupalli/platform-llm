import importlib
from configs.input_config import InputConfig
from loguru import logger


class TrainerFactory:
    @staticmethod
    def create(config: InputConfig):
        try:
            model_type = config.model.type
            class_name = f"{model_type.capitalize()}Trainer"
            module = importlib.import_module(f"trainer.{model_type}_trainer")
            class_ = getattr(module, class_name)
            logger.info(config)
            instance = class_(config)
            return instance

        except (ImportError, AttributeError):
            raise NotImplementedError(f"trainer {model_type} is not supported")

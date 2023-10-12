import importlib
from configs.data_config import DataConfig


class ReaderFactory:
    @staticmethod
    def create(config: DataConfig, exp_id: str, type: str):
        try:
            class_name = f"{type.capitalize()}Reader"
            module = importlib.import_module(f"reader.{type}_reader")
            class_ = getattr(module, class_name)
            instance = class_(config, exp_id)
            return instance

        except (ImportError, AttributeError):
            raise NotImplementedError(f"reader {type} is not supported")

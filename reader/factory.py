import importlib
from configs.data_config import DataConfig


class ReaderFactory:
    @staticmethod
    def create(config: DataConfig, exp_id: str, type_: str):
        try:
            if type_ == "seq_cls":
+               type_ = "classification"
            class_name = f"{type_.capitalize()}Reader"
            module = importlib.import_module(f"reader.{type_}_reader")
            class_ = getattr(module, class_name)
            instance = class_(config, exp_id)
            return instance

        except (ImportError, AttributeError):
            raise NotImplementedError(f"reader {type_} is not supported")

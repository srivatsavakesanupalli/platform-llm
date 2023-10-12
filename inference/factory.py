import importlib


class InferFactory:
    @staticmethod
    def create(model_type: str):
        try:
            class_name = f'{model_type.capitalize()}Infer'
            module = importlib.import_module(
                f'{model_type}_infer')
            class_ = getattr(module, class_name)
            instance = class_()
            return instance

        except (ImportError, AttributeError):
            raise NotImplementedError(
                f'Infer type {model_type} is not supported')

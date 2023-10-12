from .data_config import DataConfig
from .trainer_config import TrainerConfig
from pydantic import BaseModel


class InputConfig(BaseModel):
    proj_id: str
    exp_id: str
    username: str
    data: DataConfig
    model: TrainerConfig

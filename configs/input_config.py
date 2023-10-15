from .data_config import DataConfig
from pydantic import BaseModel


class InputConfig(BaseModel):
    proj_id: str
    exp_id: str
    username: str
    data: DataConfig
    model: BaseModel

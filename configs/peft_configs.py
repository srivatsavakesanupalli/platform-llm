from pydantic import BaseModel


class LoRAConfig(BaseModel):
    r: int
    alpha: int
    dropout: float

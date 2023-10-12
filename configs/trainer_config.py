from pydantic import BaseModel
from typing import Optional, List, Literal


class TrainerConfig(BaseModel):
    """Base configuration for specifying trainer parameters"""

    architecture: Optional[str] = None
    optimizer: Optional[str] = None
    learning_rate: float
    loss_function: Optional[str] = None
    metrics: Optional[List[str]] = None
    custom: bool
    data_augmentation: bool = False
    train_batch_size: int
    test_batch_size: int
    num_epochs: int
    threshold: float = 0.5
    type: Literal["seq_cls"]

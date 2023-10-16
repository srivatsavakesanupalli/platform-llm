from pydantic import BaseModel, Field
from typing import (
    Optional,
    Literal,
    Annotated,
    Union,
)
from configs.peft_configs import LoRAConfig as lora
from loguru import logger


class ClassificationConfig(BaseModel):
    architecture: str
    learning_rate: float
    train_batch_size: int
    test_batch_size: int
    num_epochs: int
    gradient_accumulation_steps: int
    use_peft: bool = False
    peft_method: Optional[Literal["LoRA"]] = None
    peft: lora
    threshold: float = 0.5
    type: Literal["classification"]


class NERConfig(BaseModel):
    architecture: str
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    max_steps: int
    warmup_ratio: float
    group_by_length: bool
    lr_scheduler_type: str
    train_batch_size: int
    test_batch_size: int
    num_epochs: int
    gradient_accumulation_steps: int
    save_steps: int
    logging_steps: int
    use_peft: bool = False
    peft_method: Optional[Literal["LoRA"]] = None
    peft: lora
    threshold: float = 0.5
    type: Literal["ner"]


class TrainerConfig(BaseModel):
    __root__: Annotated[
        Union[ClassificationConfig, NERConfig],
        Field(..., discriminator="type"),
    ]

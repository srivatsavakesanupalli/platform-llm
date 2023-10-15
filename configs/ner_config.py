from pydantic import BaseModel
from typing import Optional, List, Literal
from configs.peft_configs import LoRAConfig as lora


class NERConfig(BaseModel):
    """Base configuration for specifying trainer parameters"""

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
    type: str = "ner"

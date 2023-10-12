from pydantic import BaseModel
from typing import Optional, List, Literal
from configs.peft_configs import LoRAConfig as lora

class TrainerConfig(BaseModel):
    """Base configuration for specifying trainer parameters"""

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
    type: Literal["seq_cls"]

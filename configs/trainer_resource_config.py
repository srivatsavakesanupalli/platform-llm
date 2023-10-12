
from pydantic import BaseModel
from typing import Optional, List, Literal


class TrainerResourceConfig(BaseModel):
    '''Base configuration for specifying trainer parameters'''
    proj_id: str
    exp_id: str
    username: str
    min_memory: float = 3  # Memory in GB
    max_memory: float = 3.5  # Memory in GB
    min_cpu: int = 3000  # CPU in mCPU. 1000 mCPU = 1 Core
    max_cpu: int = 3100  # CPU in mCPU. 1000 mCPU = 1 Core
    backend: Literal['cpu', 'gpu']
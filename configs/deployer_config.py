from pydantic import BaseModel
from typing import Literal, Optional


class DeployerConfig(BaseModel):
    '''Base configuration required for deploying a pod to kubernetes'''
    proj_id: str
    exp_id: str
    username: str
    type: Literal['batch', 'realtime']
    batch_input: Optional[str] = None
    lifestage: Literal['Staging', 'Production']
    schedule: Optional[str] = None  # Example 0 22 * * 1-5
    min_replicas: int = 1
    max_replicas: int = 1
    min_memory: float = 0.5  # Memory in GB
    max_memory: float = 1  # Memory in GB
    min_cpu: int = 1000  # CPU in mCPU. 1000 mCPU = 1 Core
    max_cpu: int = 1100  # CPU in mCPU. 1000 mCPU = 1 Core
    stage_weight: int = 0 # Percentage of traffic to be routed to Staging
    problem_type: Optional[Literal['classification', 'segmentation', 'detection']] = None

from pydantic import BaseModel
from typing import List, Dict, Optional


class LoadTestConfig(BaseModel):
    '''Base configuration required for deploying a pod to kubernetes'''
    proj_id: str
    exp_id: str
    deploy_token: str
    ingress: str
    fail_ratio: Optional[float] = 1
    mean_response_time_ms: int
    p90_response_time_ms: int
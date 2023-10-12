from abc import ABC, abstractmethod
from exceptions.exceptions import ScheduleNotSet, BatchInputNotSet
from configs.deployer_config import DeployerConfig
from typing import List
from repo.mlflow_repo import MLFRepo
from loguru import logger
import os


class BaseDeployer(ABC):
    '''Base framework for a trainer with methods that need overriding'''

    def __init__(self, config: DeployerConfig):
        self.username = config.username
        self.image_repo = os.getenv(
            'IMAGE_REPO', '668572716132.dkr.ecr.ap-south-1.amazonaws.com')
        self.cloud_vendor = os.environ['CLOUD_VENDOR']
        self.in_cluster = int(os.getenv('IN_CLUSTER', '1'))
        self.type = config.type
        self.schedule = config.schedule
        self.image = f'{self.image_repo}/llm:{self.type}'
        self.batch_input = config.batch_input
        if self.type == 'batch':
            if not self.schedule:
                raise ScheduleNotSet('Batch deployments require a schedule')
            elif not self.batch_input:
                raise BatchInputNotSet(
                    'Batch deployments require an input for inference')
            
        self.stage_weight = config.stage_weight
        self.proj_id = config.proj_id
        self.exp_id = config.exp_id
        self.lifestage = config.lifestage
        self.min_replicas = config.min_replicas
        self.max_replicas = config.max_replicas
        self.min_memory = config.min_memory
        self.max_memory = config.max_memory
        self.min_cpu = config.min_cpu
        self.max_cpu = config.max_cpu
        self.model_name = self.get_model_name()
        self.repo = MLFRepo(self.exp_id)
        logger.info(
            f'Changing the lifestage of experiment : {self.exp_id} to {self.lifestage}')
        self.repo.exp_id_to_lifestage(
            self.model_name, self.lifestage)

    def get_model_name(self) -> str:
        return self.proj_id.lower()

    @abstractmethod
    def deploy(self) -> None:
        '''Override this method with the deployment script for various cloud providers or on-premises'''
        pass

    @abstractmethod
    def get_env(self) -> List[str]:
        '''Override this method with the environment setup function required for the deployment

        Returns:
        A list of str countaing values for the required environment values during deployment
        '''
        pass

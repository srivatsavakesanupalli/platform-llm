import os
from configs.input_config import InputConfig
from trainer.factory import TrainerFactory
from db import get_collections
from loguru import logger
import time
from utils import update_experiment_state
import torch

db_data = get_collections()
EXP = db_data['exp']
EXP_ID = os.environ['EXP_ID']
USERNAME = os.environ['USERNAME']


try:
    data = EXP.find({'exp_id': EXP_ID, 'username': USERNAME})[0]
    config = data['train_config']
    config['exp_id'] = EXP_ID
    config['proj_id'] = data['proj_id']
    config['username'] = USERNAME
except Exception as e:
    logger.info(
        f'Error loading configuration for experiment {EXP_ID} : \n {e}')
    raise e

update_experiment_state(exp_id=EXP_ID, username=USERNAME, state='PREPARING')

logger.info(f'TRAIN CONFIG : \n {config}')
logger.info('Validating configuration')
config = InputConfig(**config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Device: {device}')
logger.info('Initialising trainer')
trainer = TrainerFactory.create(config)
start_time = time.time()
logger.info('Starting training')
trainer.train()
time.sleep(60)
end_time = time.time() - start_time
logger.info(f'Training complete. Took {end_time}s')

update_experiment_state(exp_id=EXP_ID, username=USERNAME, state='TRAINED')
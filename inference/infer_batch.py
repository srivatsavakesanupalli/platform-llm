import os
from loguru import logger
from db import get_collections
from factory import InferFactory


db_data = get_collections()
DB = db_data['db']
PROJ = db_data['proj']
EXP = db_data['exp']

MODEL_NAME = os.environ["MODEL_NAME"]
EXP_ID = os.environ["EXP_ID"]

logger.info(f'Fetching configuration for {MODEL_NAME}')
data = EXP.find({'exp_id': EXP_ID})[0]
config = data['train_config']
logger.info(f'TRAIN CONFIG : \n {config}')

ENGINE = InferFactory.create(config['model']['type'])
ENGINE.infer_batch()

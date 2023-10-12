import psutil
import os
import time
from loguru import logger
from db import get_collections

db_data = get_collections()
DB = db_data['db']
COLLECTION = os.getenv('MONGO_PROJ_COLLECTION', 'projects')#os.environ['COLLECTION_NAME']
collection = DB[COLLECTION]

MODEL_NAME = os.environ['MODEL_NAME']
STATE = os.environ['STATE']


def log_metrics(metrics_dict):
    metrics_dict['cpu_usage'] = psutil.cpu_percent()
    metrics_dict['virtual_memory_usage'] = psutil.virtual_memory().percent
    metrics_dict['swap_memory_usage'] = psutil.swap_memory().percent
    metrics_dict['model_name'] = MODEL_NAME
    metrics_dict['model_version'] = STATE
    metrics_dict['timestamp'] = int(time.time()*1000)
    logger.info(f'Inference Metrics : {metrics_dict}')
    collection.insert_one(metrics_dict)
    logger.info('Metrics recorded')

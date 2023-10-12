from pymongo import MongoClient
import pymongo
from loguru import logger
from datetime import datetime
from typing import Optional
import os
from uuid import uuid4
import time
from fastapi import HTTPException
from sqlmodel import Field, Session, SQLModel, create_engine,select


MONGO_URL = os.getenv('MONGO_URL', 'localhost:27017')
MONGO_DB = os.getenv('MONGO_DB', 'default')
PROJ_COLLECTION = os.getenv('MONGO_PROJ_COLLECTION', 'projects')
EXP_COLLECTION = os.getenv('MONGO_EXP_COLLECTION', 'experiments')
POSTGRES_URL = os.getenv('POSTGRES_URL', 'postgresql://postgres:f3lt0J0fRl@localhost:5432/postgres')
# POSTGRES_URL = 'postgresql://postgres:f3lt0J0fRl@postgresdb-postgresql.postgres.svc.cluster.local:5432/postgres'
SQL_ENGINE = create_engine(POSTGRES_URL, encoding='utf-8')

class CVParams(SQLModel, table=True, extend_existing=True):
    id: Optional[str] = Field(primary_key=True)
    run_id: str
    experiment_name: str
    param_name: str
    param_value: Optional[float] = None
    step: Optional[int] = 0
    timestamp: datetime = Field(default=datetime.now())
    metric_type: Optional[str] = None

class CVAugmentations(SQLModel, table=True, extend_existing=True):
    id: Optional[str] = Field(primary_key=True)
    run_id: str
    experiment_name: str
    augmentation_type: str
    param_name: str
    param_value: str
    timestamp: datetime = Field(default=datetime.now())

try:
    SQLModel.metadata.create_all(SQL_ENGINE)
except Exception as e:
    logger.warning(f'Table creation did not succeed with the following error: {e}')

def push_to_postgres(params, run_id, experiment_id, step=0, metric_type=None, engine=SQL_ENGINE):
    timestamp = datetime.now()
    with Session(engine) as session:
        for param in params:
            row_id = uuid4().hex[:8]
            instance = CVParams(id=row_id, run_id=run_id, experiment_name=experiment_id, param_name=param, param_value=params[param], step=step, timestamp=timestamp, metric_type=metric_type)
            session.add(instance)
        session.commit()

def push_aug_to_postgres(params, run_id, experiment_id, augmentation_type, engine=SQL_ENGINE):
    timestamp = datetime.now()
    with Session(engine) as session:
        for param in params:
            row_id = uuid4().hex[:8]
            instance = CVAugmentations(
                id=row_id,
                run_id=run_id,
                experiment_name=experiment_id,
                augmentation_type=augmentation_type,
                param_name=param,
                param_value=params[param],
                timestamp=timestamp
            )
            session.add(instance)
        session.commit()

def cleanup_postgres(experiment_id, engine=SQL_ENGINE):
    logger.warning(f'Cleaning up CV data for the exp : {experiment_id}')
    with Session(engine) as session:
        statement = select(CVParams).where(CVParams.experiment_name == experiment_id)
        results = session.exec(statement)
        all_rows = results.all()
        logger.info(f'Removing {len(all_rows)} records')
        for result in all_rows:
            session.delete(result)
        session.commit()

        statement = select(CVParams).where(CVAugmentations.experiment_name == experiment_id)
        results = session.exec(statement)
        all_rows = results.all()
        logger.info(f'Removing {len(all_rows)} records')
        for result in all_rows:
            session.delete(result)
        session.commit()

def read_from_postgres(metric, run_id, experiment_id, engine=SQL_ENGINE):
    with Session(engine) as session:
        statement = select(CVParams).where(CVParams.experiment_name == experiment_id, \
                                          CVParams.run_id == run_id, \
                                          CVParams.param_name == metric)
        results = session.exec(statement)
        all_rows = results.all()
        return all_rows

def get_collections(uri: str = MONGO_URL):
    logger.info('Initialising MongoDB')
    mongo_conf = {}
    uri_data = pymongo.uri_parser.parse_uri(uri)
    hostslist = [f'mongodb://{x[0]}:{x[1]}' if not ix else f'{x[0]}:{x[1]}' for ix, x in enumerate(uri_data['nodelist'])]
    mongo_conf = {
        **uri_data['options'],
        'host': hostslist,
        'username': uri_data['username'],
        'password': uri_data['password'],
    }
    conn = MongoClient(**mongo_conf)
    db = conn[MONGO_DB]
    exp_collection = db[EXP_COLLECTION]
    proj_collection = db[PROJ_COLLECTION]
    # datasets_collection = db[DATASET_COLLECTION]
    logger.info('Mongo Client Initialised')
    return {'exp': exp_collection, 
            'proj': proj_collection, 
            # 'datasets': datasets_collection,
            'db': db}


def get_deploy_token(username: str):
    try:
        users_collection = get_collections()['db']['user_data']
        stored_token = list(users_collection.find(
            {'username': username}))[0]['deploy_token']
        return {'deploy_token': stored_token}
    except Exception:
        raise HTTPException(
            status_code=403, detail="Please create an access token.")


try:
    username = os.environ['USERNAME']
    STORED_TOKEN = get_deploy_token(username)['deploy_token']
except Exception as e:
    logger.warning(
        f'Error fetching Deployment Token. Ignore this if you are in the training envirnoment \n {e}')


def validate_deploy_token(token: str):
    if STORED_TOKEN != token:
        raise HTTPException(status_code=403, detail="Invalid deploy token")
    return True

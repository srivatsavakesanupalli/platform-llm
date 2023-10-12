from db import get_collections
from uuid import uuid4
from exceptions.exceptions import UnknownExperiment
import zipfile
from PIL import Image
import torch
import boto3
from collections import defaultdict
import json
from botocore.exceptions import ClientError
import numpy as np
import os
import functools
import torch
import mlflow
import time
from multiprocessing.pool import ThreadPool
from loguru import logger
from fastapi import HTTPException
from kubernetes import config, client
from typing import Literal
from db import get_collections
from repo.mlflow_repo import MLFRepo

EXP = get_collections()['exp']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def upload_file_to_s3(bucket_name, source_file_path, destination_object_key):
    s3_client = boto3.client('s3',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                region_name="ap-south-1")
    try:
        response = s3_client.upload_file(source_file_path, bucket_name, destination_object_key)
    except ClientError as e:
        logger.error(e)

def list_s3_objects(exp_id, run_id):
    s3_client = boto3.client('s3',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                region_name="ap-south-1")
    try:
        response = s3_client.list_objects_v2(Bucket='nocode-awone',
                                     Prefix=f'platform_llm/{exp_id}/{run_id}')
        return [d['Key'] for d in response['Contents']]
    except ClientError as e:
        logger.error(e)
        return []

def generate_preSignedURLs(objects):
    s3_client = boto3.client('s3',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                region_name="ap-south-1")
    urls = []
    try:
        for object_key in objects:
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': 'nocode-awone',
                    'Key': object_key
                },
                ExpiresIn=3600*3
            )
            urls.append(url)
    except ClientError as e:
        logger.error(e)
    return urls

def mapPreSignedURLs(objects, urls, threshold, exp_id, run_id):
    repo = MLFRepo(exp_id)
    path = repo.download_artifact(run_id, 'misclassifications/wrong.json')
    with open(path, 'r') as f:
        d = json.load(f)
    mapping = defaultdict(list)
    for class_ in d[threshold]:
        for file in d[threshold][class_]:
            file_name = '_'.join(file.split(os.sep)[-2:])
            if f'platform_llm/{exp_id}/{run_id}/{file_name}' in objects:
                mapping[class_].append(urls[objects.index(f'platform_llm/{exp_id}/{run_id}/{file_name}')])
    return dict(mapping)


def get_proj_id(exp_id: str) -> str:
    exp_collection = get_collections()['exp']
    exp_data = exp_collection.find({'exp_id': exp_id})[0]
    if not exp_data:
        raise UnknownExperiment(f'Experiment {exp_id} does not exist')
    return exp_data['proj_id']

def update_experiment_state(exp_id: str, username: str, state: Literal['CREATED', 'PREPARED', 'TRAINING', 'TRAINED', 'DEPLOYED']) -> None:
    '''Helper method to change the state of an experiment

    Args:
        exp_id: Current data from MongoDB
        username: Username who the exp belongs to
        state: State to which it should be updated to
    '''
    logger.info(f'Updating experiment {exp_id} to {state} status')
    EXP.update_one({'exp_id': exp_id, 'username': username}, {"$set": {
                   "status": state, "ts": int(time.time()*1000)}})
    logger.info(f'{exp_id} Update complete')

def generate_proj_id() -> str:
    proj_collection = get_collections()['proj']
    try:
        exp_data = proj_collection.find({}, {'proj_id': 1}).sort(
            "created_ts", -1).limit(1)[0]
        logger.info(exp_data)
    except Exception as e:
        logger.info(e)
        return 'PRJ-100000'
    current_id = exp_data['proj_id']
    logger.info(exp_data['proj_id'])
    current_sequence = int(current_id.split('-')[1])
    next_in_sequence = current_sequence + 1
    new_id = f'PRJ-{next_in_sequence}'
    return new_id


def generate_exp_id(project_id: str) -> str:
    hash_ = uuid4().hex[:8]
    return f'{project_id}-{hash_}'


def unzip_file(zip_filepath: str, extract_location: str) -> None:
    """
    Function to unzip a file.

    Args:
    zip_filepath : str : location of the zip file
    extract_location : str : location to extract the contents of the zip file

    Returns:
    None
    """
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_location)

def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)

    vec1_length = np.linalg.norm(vec1)
    vec2_length = np.linalg.norm(vec2)

    similarity = dot_product / (vec1_length * vec2_length)

    return 1-similarity


def get_ingress_model_name(model_name: str) -> str:
    '''Method to get the deployed ingress from kubernetes given a model name

    Args:
        model_name: Model Name for which the deployment was triggered

    Returns:
        Ingress URL for the deployment
    '''
    namespace = f'{model_name}-infer'
    ingress_name = f'{namespace}-ingress'

    in_cluster = int(os.getenv('IN_CLUSTER', '1'))

    if in_cluster:
        config.load_incluster_config()
    else:
        config.load_kube_config()

    network_api = client.NetworkingV1Api()
    try:
        service = network_api.read_namespaced_ingress(ingress_name, namespace)
        logger.info(service.status.load_balancer.ingress)
        ingress = service.status.load_balancer.ingress[0].ip
        ingress = f'{ingress}/{model_name}'
    except Exception as e:
        logger.info(f'Error while fetching ingress: {e}')
        ingress = ''

    return ingress


def get_external_ip(model_name: str):
    '''Method to get the deployed ingress from kubernetes given a model name

    Args:
        model_name: Model Name for which the deployment was triggered

    Returns:
        Ingress URL for the deployment
    '''
    namespace = f'{model_name}-infer'
    service_name = f'{namespace}-service'

    in_cluster = int(os.getenv('IN_CLUSTER', '1'))

    if in_cluster:
        config.load_incluster_config()
    else:
        config.load_kube_config()

    v1 = client.CoreV1Api()
    external_ip = ''
    try:
        service = v1.read_namespaced_service(
            name=service_name, namespace=namespace)
        logger.info(f'Service: {service}')
        if service.spec.type == "LoadBalancer" and service.status.load_balancer.ingress:
            external_ip = service.status.load_balancer.ingress[0].ip
    except Exception as e:
        logger.info(f'Error while fetching external IP: {e}')
        external_ip = ''

    return external_ip

def get_latest_run_id(exp_id):
    client = mlflow.MlflowClient()
    model_name = ('-'.join(exp_id.split('-')[0:2])).lower()
    versions = client.get_latest_versions(model_name)
    versions = sorted(versions, key=lambda x: x.creation_timestamp)
    run_id = versions[-1].run_id
    return run_id

def get_metrics(run_id, metric_key):
    client = mlflow.MlflowClient()
    d = client.get_metric_history(run_id=run_id, key=metric_key)
    if len(d)>0:
        d = {f'{measure.step}': measure.value for measure in d}
        return d
    else:
        return {}

def get_run_status(exp_id: str):
    '''Helper method to get the runs and status from mlflow in a given experiment

    Args:
        exp_id : Experiment name in mlflow
    '''
    try:
        exp_id = mlflow.get_experiment_by_name(exp_id).experiment_id
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f'Experiment not yet initialised {e}')
    runs = mlflow.search_runs(exp_id)
    # Look at runs which dont have a run_version tag yet
    if 'tags.run_version' in runs.columns:
        runs = runs[runs['tags.run_version'].isna()]
    if len(runs):
        runs = runs.sort_values('start_time', ascending=False).loc[:, ['run_id', 'status']]
        return runs.to_dict(orient='records')
    else:
        return []

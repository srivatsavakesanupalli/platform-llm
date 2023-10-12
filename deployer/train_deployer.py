import os
from kubernetes import client, config, utils
from configs.trainer_resource_config import TrainerResourceConfig
from kubernetes.client.rest import ApiException
from loguru import logger
import yaml
import time


class TRAINDeployer():
    '''Deployer for generic k8s clusters'''

    def __init__(self, config: TrainerResourceConfig):
        self.cloud_vendor = os.environ['CLOUD_VENDOR']
        self.in_cluster = int(os.getenv('IN_CLUSTER', '1'))
        self.exp_id = config.exp_id
        self.model_name = config.proj_id.lower()
        self.username = config.username
        self.min_memory = config.min_memory
        self.max_memory = config.max_memory
        self.min_cpu = config.min_cpu
        self.max_cpu = config.max_cpu
        self.backend = config.backend
        self.image = f"{os.getenv('IMAGE_REPO', '668572716132.dkr.ecr.ap-south-1.amazonaws.com')}/platform-cv:latest"
        self.mlflow_uri,\
            self.mongo,\
            self.db = self.get_env()

    def get_env(self):
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost')
        mongo_uri = os.getenv('MONGO_URL', 'mongodb://localhost:27017')
        db_name = os.getenv('MONGO_DB', 'default')
        return [mlflow_uri, mongo_uri, db_name]

    def replace_content(self, yaml_content: str) -> str:
        '''Replace templated values in the yaml with the actual values from the object

        Args:
            yaml_content : Content of the template yaml file in string

        Returns:
            Replaced yaml content
        '''

        new_content = yaml_content.replace('CONTAINER_IMAGE', self.image)\
            .replace('ACCESSKEY_AWS', os.getenv('AWS_ACCESS_KEY_ID', ""))\
            .replace('SECRETKEY_AWS', os.getenv('AWS_SECRET_ACCESS_KEY', ""))\
            .replace('ACCESSKEY_AZURE', os.getenv('AZURE_STORAGE_ACCESS_KEY', ""))\
            .replace('SECRETKEY_AZURE', os.getenv('AZURE_STORAGE_CONNECTION_STRING', ""))\
            .replace('NAMESPACE', self.model_name + '-train')\
            .replace('MLFLOW_TRACKING_URI_VALUE', self.mlflow_uri)\
            .replace('MIN_CPU', f"{self.min_cpu}m")\
            .replace('MAX_CPU', f"{self.max_cpu}m")\
            .replace('MIN_MEMORY', f"{self.min_memory}Gi")\
            .replace('MAX_MEMORY', f"{self.max_memory}Gi")\
            .replace('POSTGRES_URL_VALUE', os.getenv('POSTGRES_URL'))\
            .replace('MONGOURL', self.mongo)\
            .replace('DBNAME', self.db)\
            .replace('EXP_ID_VALUE', self.exp_id)\
            .replace('USERNAME_VALUE', self.username)

        return new_content

    def create_yaml(self):
        '''Write the updated yaml file onto disk'''
        os.makedirs('yamls', exist_ok=True)
        if self.backend == 'cpu':
            yaml_path = 'templates/train_cpu.yaml'
        elif self.backend == 'gpu':
            yaml_path = 'templates/train_gpu.yaml'
        yaml_content = open(yaml_path, 'r').read()
        yaml_content = self.replace_content(yaml_content)
        file_path = f'yamls/train_{self.exp_id}.yaml'
        with open(file_path, 'w') as f:
            f.write(yaml_content)
        return file_path, yaml_content

    def deploy(self):
        '''Create the yaml from template for the deployment and apply using k8s API
        '''
        logger.info('Creating yamls')
        yaml_file, yaml_content = self.create_yaml()
        logger.info('YAMLS created')
        logger.info(yaml_content)
        logger.info('Deployment started')
        self.create_or_update_k8s_resource_from_yaml(yaml_file)
        logger.info('Deployment successful. Cleaning up yaml')
        os.remove(yaml_file)
        logger.info('All Done!!!!')

    def create_or_update_k8s_resource_from_yaml(self, yaml_file_path):
        if self.in_cluster:
            config.load_incluster_config()
        else:
            config.load_kube_config()

        # Read YAML file and create/update the Kubernetes objects.
        with open(yaml_file_path, 'r') as f:
            resource_list = list(yaml.safe_load_all(f))

        # Iterate through each resource in the YAML file.
        for resource in resource_list:
            if resource is not None:
                try:
                    # Check the resource kind and create/update it accordingly.
                    if resource['kind'] == 'Namespace':
                        api = client.CoreV1Api()
                        try:
                            api.create_namespace(body=resource)
                            logger.info(
                                f"Namespace {resource['metadata']['name']} created.")
                        except ApiException as e:
                            if e.status == 409:
                                api.patch_namespace(
                                    name=resource['metadata']['name'], body=resource)
                                logger.info(
                                    f"Namespace {resource['metadata']['name']} updated.")
                            else:
                                raise e

                    elif resource['kind'] == 'Job':
                        api = client.BatchV1Api()
                        try:
                            api.delete_namespaced_job(name=resource['metadata']['name'],
                                                      namespace=resource['metadata']['namespace'],
                                                      propagation_policy="Foreground")
                            while True:
                                try:
                                    api.read_namespaced_job(name=resource['metadata']['name'],
                                                            namespace=resource['metadata']['namespace'])
                                    time.sleep(1)
                                except ApiException as e:
                                    if e.status == 404:
                                        break
                                    else:
                                        raise
                            logger.info(
                                f"Deleted existing job '{resource['metadata']['name']}' in namespace '{resource['metadata']['namespace']}'")
                        except Exception as e:
                            if e.status == 404:
                                logger.info(
                                    'Nothing to delete. Creating a new job')
                        api.create_namespaced_job(
                            namespace=resource['metadata']['namespace'], body=resource)

                    # Add more conditions for other resource types as needed.

                except Exception as e:
                    logger.info(
                        f"Failed to create/update {resource['kind']} {resource['metadata']['name']} - {e}")
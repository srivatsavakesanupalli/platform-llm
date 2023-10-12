from .base_deployer import BaseDeployer
import os
from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException
from loguru import logger
import yaml
from datetime import datetime


class K8SDeployer(BaseDeployer):
    '''Deployer for generic k8s clusters'''

    def __init__(self, config):
        super().__init__(config)
        self.mlflow_uri,\
            self.mongo,\
            self.db,\
            self.collection = self.get_env()

    def get_env(self):
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost')
        mongo_uri = os.getenv('MONGO_URL', 'mongodb://localhost:27017')
        db_name = os.getenv('MONGO_DB', 'default')
        collection = os.getenv('MONGO_LOG_COLLECTION', 'log-monitor')
        return [mlflow_uri, mongo_uri, db_name, collection]

    def replace_content(self, yaml_content: str) -> str:
        '''Replace templated values in the yaml with the actual values from the object

        Args:
            yaml_content : Content of the template yaml file in string

        Returns:
            Replaced yaml content
        '''
        timestamp = datetime.utcnow().astimezone()
        timezone_offset = timestamp.strftime('%z')
        timezone_offset_formatted = f"{timezone_offset[:-2]}:{timezone_offset[-2:]}"
        formatted_time = timestamp.strftime(
            f'%Y-%m-%dT%H:%M:%S{timezone_offset_formatted}')

        new_content = yaml_content.replace('CONTAINER_IMAGE', self.image)\
            .replace('ACCESSKEY_AWS', os.getenv('AWS_ACCESS_KEY_ID', ""))\
            .replace('SECRETKEY_AWS', os.getenv('AWS_SECRET_ACCESS_KEY', ""))\
            .replace('ACCESSKEY_AZURE', os.getenv('AZURE_STORAGE_ACCESS_KEY', ""))\
            .replace('SECRETKEY_AZURE', os.getenv('AZURE_STORAGE_CONNECTION_STRING', ""))\
            .replace('NAMESPACE', self.model_name + '-infer')\
            .replace('MODEL_NAME_VALUE', self.model_name)\
            .replace('MLFLOW_TRACKING_URI_VALUE', self.mlflow_uri)\
            .replace('MIN_REPLICAS', str(self.min_replicas))\
            .replace('MAX_REPLICAS', str(self.max_replicas))\
            .replace('MIN_CPU', f"{self.min_cpu}m")\
            .replace('MAX_CPU', f"{self.max_cpu}m")\
            .replace('MIN_MEMORY', f"{self.min_memory}Gi")\
            .replace('MAX_MEMORY', f"{self.max_memory}Gi")\
            .replace('MONGOURL', self.mongo)\
            .replace('DBNAME', self.db)\
            .replace('COLLECTIONNAME', self.collection)\
            .replace('EXP_ID_VALUE', self.exp_id)\
            .replace('TIMESTAMP_VALUE', formatted_time)\
            .replace('USERNAME_VALUE', self.username)
        if self.type == 'batch':
            new_content = new_content.replace('SCHEDULE', self.schedule)\
                .replace('BATCH_INPUT_VALUE', self.batch_input)\
                .replace('VENDOR_VALUE', self.cloud_vendor)

        return new_content

    def create_yaml(self):
        '''Write the updated yaml file onto disk'''
        os.makedirs('yamls', exist_ok=True)
        if self.type == 'realtime':
            yaml_path = 'templates/rti.yaml'
        else:
            yaml_path = 'templates/batch.yaml'
        yaml_content = open(yaml_path, 'r').read()
        yaml_content = self.replace_content(yaml_content)
        file_path = f'yamls/{self.type}_{self.exp_id}.yaml'
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
                    elif resource['kind'] == 'Pod':
                        api = client.CoreV1Api()
                        api.create_namespaced_pod(
                            namespace=resource['metadata']['namespace'], body=resource)
                        logger.info(
                            f"Pod {resource['metadata']['name']} created.")
                    elif resource['kind'] == 'Deployment':
                        api = client.AppsV1Api()
                        try:
                            api.create_namespaced_deployment(
                                namespace=resource['metadata']['namespace'], body=resource)
                            logger.info(
                                f"Deployment {resource['metadata']['name']} created.")
                        except ApiException as e:
                            if e.status == 409:  # HTTP 409 Conflict indicates the resource already exists.
                                api.patch_namespaced_deployment(name=resource['metadata']['name'],
                                                                namespace=resource['metadata']['namespace'],
                                                                body=resource)
                                logger.info(
                                    f"Deployment {resource['metadata']['name']} updated.")
                            else:
                                raise e

                    elif resource['kind'] == 'Service':
                        api = client.CoreV1Api()
                        try:
                            api.create_namespaced_service(
                                namespace=resource['metadata']['namespace'], body=resource)
                            logger.info(
                                f"Service {resource['metadata']['name']} created.")
                        except ApiException as e:
                            if e.status == 409:
                                api.patch_namespaced_service(name=resource['metadata']['name'],
                                                             namespace=resource['metadata']['namespace'],
                                                             body=resource)
                                logger.info(
                                    f"Service {resource['metadata']['name']} updated.")
                            else:
                                raise e
                    elif resource['kind'] == 'Ingress':
                        api = client.NetworkingV1Api()
                        try:
                            api.create_namespaced_ingress(
                                namespace=resource['metadata']['namespace'], body=resource)
                            logger.info(
                                f"Ingress {resource['metadata']['name']} created.")
                        except ApiException as e:
                            if e.status == 409:
                                api.patch_namespaced_ingress(name=resource['metadata']['name'],
                                                             namespace=resource['metadata']['namespace'],
                                                             body=resource)
                                logger.info(
                                    f"Ingress {resource['metadata']['name']} updated.")
                            else:
                                raise e
                    elif resource['kind'] == 'HorizontalPodAutoscaler':
                        api = client.AutoscalingV2Api()
                        try:
                            api.create_namespaced_horizontal_pod_autoscaler(
                                namespace=resource['metadata']['namespace'], body=resource)
                            logger.info(
                                f"HPA {resource['metadata']['name']} created.")
                        except ApiException as e:
                            if e.status == 409:
                                api.patch_namespaced_horizontal_pod_autoscaler(name=resource['metadata']['name'],
                                                                               namespace=resource['metadata']['namespace'],
                                                                               body=resource)
                                logger.info(
                                    f"HPA {resource['metadata']['name']} updated.")
                            else:
                                raise e
                    elif resource['kind'] == 'CronJob':
                        api = client.BatchV1Api()
                        try:
                            api.create_namespaced_cron_job(
                                namespace=resource['metadata']['namespace'], body=resource)
                            logger.info(
                                f"CronJob {resource['metadata']['name']} created.")
                        except ApiException as e:
                            if e.status == 409:
                                api.patch_namespaced_cron_job(name=resource['metadata']['name'],
                                                              namespace=resource['metadata']['namespace'],
                                                              body=resource)
                                logger.info(
                                    f"CronJob {resource['metadata']['name']} updated.")
                            else:
                                raise e

                    # Add more conditions for other resource types as needed.

                except Exception as e:
                    logger.info(
                        f"Failed to create/update {resource['kind']} {resource['metadata']['name']} - {e}")

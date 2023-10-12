import os
from configs.loadtest_config import LoadTestConfig
from kubernetes import client, config, utils
from kubernetes.client.rest import ApiException
from loguru import logger
import yaml
import json
import time


class LOCUSTDeployer:
    '''Deployer for Locust/Load Testing Services for generic k8s clusters'''

    def __init__(self, config: LoadTestConfig):
        self.deploy_token = config.deploy_token
        self.in_cluster = int(os.getenv('IN_CLUSTER', '1'))
        self.host_url = config.ingress
        self.exp_id = config.exp_id
        self.model_name = config.proj_id.lower()
        self.mongo_uri = os.getenv('MONGO_URL', 'mongodb://localhost:27017')
        self.db_name = os.getenv('MONGO_DB', 'default')
        self.fail_ratio = config.fail_ratio
        self.avg_resp_time = config.mean_response_time_ms
        self.p90_resp_time = config.p90_response_time_ms
        self.image = f"{os.getenv('IMAGE_REPO', '668572716132.dkr.ecr.ap-south-1.amazonaws.com')}/locustpymongo:latest"

    def replace_content(self, py_content: str, yaml_content: str) -> tuple:
        '''Replace templated values in the yaml with the actual values from the object

        Args:
            py_content : Content of the python file for loadtesting
            yaml_content : Content of the template yaml file in string

        Returns:
            Replaced .py and yaml content
        '''
        py_content = py_content.replace('MONGO_URL_VALUE', self.mongo_uri)\
            .replace('MONGO_DB_VALUE', self.db_name)\
            .replace('EXP_ID_VALUE', self.exp_id)\
            .replace('FAIL_RATIO_VALUE', str(self.fail_ratio))\
            .replace('AVG_RESPONSE_TIME_VALUE', str(self.avg_resp_time))\
            .replace('P90_RESPONSE_TIME_VALUE', str(self.p90_resp_time))\
            .replace('DEPLOY_TOKEN_VALUE', self.deploy_token)

        yaml_content = yaml_content.replace('MODEL_NAME_VALUE', self.model_name)\
            .replace('EXP_ID_VALUE', self.exp_id)\
            .replace('HOSTURL', self.host_url)\
            .replace('CONTAINER_IMAGE', self.image)

        return py_content, yaml_content

    def create_yaml(self):
        '''Write the updated yaml file onto disk'''
        os.makedirs('yamls', exist_ok=True)
        yaml_path = 'templates/loadtest.yaml'
        locustfile_path = 'templates/locustfile.py'
        py_content = open(locustfile_path, 'r').read()
        yaml_content = open(yaml_path, 'r').read()
        py_content, yaml_content = self.replace_content(
            py_content, yaml_content)
        yaml_file_path = f'yamls/loadtest_{self.exp_id}.yaml'
        py_file_path = f'yamls/loadtest_{self.model_name}.py'
        with open(yaml_file_path, 'w') as f:
            f.write(yaml_content)
        with open(py_file_path, 'w') as f:
            f.write(py_content)

        return {'py': {'file_path': py_file_path, 'content': py_content},
                'yaml': {'file_path': yaml_file_path, 'content': yaml_content}}

    def create_configmap_object(self, file, name, namespace):
        # Configureate ConfigMap metadata
        metadata = client.V1ObjectMeta(
            name=name,
            namespace=namespace,
        )
        # Get File Content
        with open(file, 'r') as f:
            file_content = f.read()
        # Instantiate the configmap object
        fname = os.path.basename(file)
        configmap = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            data={fname: file_content},
            metadata=metadata
        )

        return configmap

    def create_configmap(self, configmap, namespace):
        if self.in_cluster:
            config.load_incluster_config()
        else:
            config.load_kube_config()
        api_instance = client.CoreV1Api()
        try:
            api_response = api_instance.create_namespaced_config_map(
                namespace=namespace,
                body=configmap,
                pretty='pretty',
            )
            logger.info(api_response)
        except ApiException as e:
            if e.status == 409:
                api_instance.patch_namespaced_config_map(namespace=namespace,
                                                         body=configmap,
                                                         name=self.model_name)
                logger.info(
                    "ConfigMap updated.")
            else:
                raise e

    def deploy(self):
        '''Create the yaml from template for the deployment and apply using k8s API
        '''
        logger.info('Creating yaml and locustfile')
        created_data = self.create_yaml()
        logger.info('YAML created')
        logger.info(created_data['yaml']['content'])
        logger.info('locustfile created')
        logger.info(created_data['py']['content'])
        logger.info('Deployment started')
        logger.info('Creating the configmap')
        configmap = self.create_configmap_object(
            created_data['py']['file_path'], self.model_name, 'locust')
        self.create_configmap(configmap, 'locust')
        logger.info('Configmap created. Deploying the Custom resource')
        self.apply_locusttest_cr(created_data['yaml']['file_path'], 'locust')
        logger.info('Deployment successful removing temporary files')
        os.remove(created_data['py']['file_path'])
        os.remove(created_data['yaml']['file_path'])
        logger.info('All Done!!!!')

    def apply_locusttest_cr(self, yaml_file_path, namespace):
        if self.in_cluster:
            config.load_incluster_config()
        else:
            config.load_kube_config()
        try:
            group = "locust.io"
            version = "v1"
            plural = "locusttests"
            api = client.CustomObjectsApi()
            with open(yaml_file_path, 'r') as f:
                crd_yaml = f.read()
            api.create_namespaced_custom_object(group=group,
                                                version=version,
                                                namespace=namespace,
                                                plural=plural,
                                                body=yaml.safe_load(crd_yaml))
        except ApiException as e:
            if e.status == 409:
                api.delete_namespaced_custom_object(group=group,
                                                    version=version,
                                                    namespace=namespace,
                                                    plural=plural,
                                                    name=f'{self.model_name}.test')

                time.sleep(5)

                api.create_namespaced_custom_object(group=group,
                                                    version=version,
                                                    namespace=namespace,
                                                    plural=plural,
                                                    body=yaml.safe_load(crd_yaml))
            else:
                raise e

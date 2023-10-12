from .base_deployer import BaseDeployer
import boto3
import os


class AWSDeployer(BaseDeployer):
    '''Deployer for AWS'''

    def __init__(self, config):
        super().__init__(config)
        self.access_key,\
            self.secret_key,\
            self.region,\
            self.code_build_project_base,\
            self.cluster_name,\
            self.arn,\
            self.mlflow_uri,\
            self.mongo,\
            self.db,\
            self.collection = self.get_env()
        if self.type == 'realtime':
            self.code_build_project_name = self.code_build_project_base
        else:
            self.code_build_project_name = f'{self.code_build_project_base}-batch'

    def get_env(self):
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region = os.getenv('AWS_DEFAULT_REGION')
        code_build_project_base = os.getenv('CODE_BUILD_PROJECT')
        cluster_name = os.getenv('EKS_CLUSTER_NAME')
        arn = os.getenv('EKS_KUBECTL_ROLE_ARN')
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost')
        mongo_uri = os.getenv('MONGO_URL', 'mongodb://localhost:27017')
        db_name = os.getenv('MONGO_DB', 'default')
        collection = os.getenv('MONGO_LOG_COLLECTION', 'log-monitor')
        return [access_key, secret_key, region, code_build_project_base, cluster_name, arn, mlflow_uri, mongo_uri, db_name, collection]

    def deploy(self):
        client = boto3.client('codebuild', region_name=self.region)
        # Start the deployment of Real Time Inference
        namespace = self.model_name + '-infer'
        overrides = [
            {"name": "EKS_CLUSTER_NAME",
             "value": self.cluster_name, "type": "PLAINTEXT"},
            {"name": "NAMESPACE", "value": namespace,
             "type": "PLAINTEXT"},
            {"name": "REPOSITORY_URI",
             "value": self.image, "type": "PLAINTEXT"},
            {"name": "EKS_KUBECTL_ROLE_ARN",
             "value": self.arn, "type": "PLAINTEXT"},
            {"name": "MIN_REPLICAS", "value": str(self.min_replicas),
             "type": "PLAINTEXT"},
            {"name": "MAX_REPLICAS", "value": str(self.max_replicas),
             "type": "PLAINTEXT"},
            {"name": "MIN_CPU", "value": f"{self.min_cpu}m",
             "type": "PLAINTEXT"},
            {"name": "MAX_CPU", "value": f"{self.max_cpu}m",
             "type": "PLAINTEXT"},
            {"name": "MIN_MEMORY", "value": f"{self.min_memory}Gi",
             "type": "PLAINTEXT"},
            {"name": "MAX_MEMORY", "value": f"{self.max_memory}Gi",
             "type": "PLAINTEXT"},
            {"name": "ACCESSKEY", "value": self.access_key,
             "type": "PLAINTEXT"},
            {"name": "SECRETKEY", "value": self.secret_key,
             "type": "PLAINTEXT"},
            {"name": "MODEL_NAME", "value": self.model_name,
             "type": "PLAINTEXT"},
            {"name": "EXP_ID_VALUE", "value": self.exp_id,
             "type": "PLAINTEXT"},
            {"name": "MLFLOW_TRACKING_URI_VALUE",
             "value": self.mlflow_uri, "type": "PLAINTEXT"},
            {"name": "MONGOURL", "value": self.mongo,
             "type": "PLAINTEXT"},
            {"name": "DBNAME",
             "value": self.db, "type": "PLAINTEXT"},
            {"name": "COLLECTIONNAME",
             "value": self.collection, "type": "PLAINTEXT"}
        ]
        if self.type == 'batch':
            overrides += [{"name": "SCHEDULE",
                           'value': self.schedule, "type": 'PLAINTEXT'},
                          {"name": "BATCH_INPUT_VALUE",
                           'value': self.batch_input, "type": 'PLAINTEXT'}]
        client.start_build(
            projectName=self.code_build_project_name,
            gitCloneDepthOverride=1,
            artifactsOverride={
                'type': 'CODEPIPELINE',
            },
            environmentVariablesOverride=overrides
        )

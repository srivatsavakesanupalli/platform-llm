from .base_storage import BaseStorage
import boto3
from botocore.exceptions import ClientError
from loguru import logger
import os
from exceptions.exceptions import EnvNotSet
from datetime import datetime, timezone


class AWSStorage(BaseStorage):
    def __init__(self, filepath: str, object_path: str) -> None:
        super().__init__(filepath, object_path)
        logger.info(os.environ["AWS_ACCESS_KEY_ID"])
        logger.info(os.environ["AWS_SECRET_ACCESS_KEY"])
        try:
            self.bucket_name = os.getenv('BUCKET_NAME', 'nocode-awone')
        except KeyError:
            raise EnvNotSet('BUCKET_NAME is not configured')

        assert object_path.startswith('s3'), "S3 input path expected"

        self.object_key = '/'.join(object_path.split('/')[3:])

    def upload(self):
        if self.object_key is None:
            object_name = os.path.basename(self.filepath)
        else:
            object_name = self.object_key
        # Upload the file
        s3_client = boto3.client('s3')
        try:
            _ = s3_client.upload_file(
                self.filepath, self.bucket_name, object_name)
        except ClientError as e:
            logger.error(e)
            return False
        return True

    def download(self):
        s3_client = boto3.resource('s3')
        modified = self.is_modified()
        # os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        logger.info(f"downloading to {self.filepath}")
        download_bucket_name = self.object_path.split('/')[2]
        if not os.path.isfile(self.filepath) or modified:
            try:
                experiment_folder = os.path.dirname(self.filepath)
                os.makedirs(experiment_folder, exist_ok=True)
                s3_client.Bucket(download_bucket_name).download_file(
                    self.object_key, self.filepath)
                logger.info(f'{self.filepath} Downloaded')
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logger.info(f"{self.filepath} does not exist in s3")
                    return False
                else:
                    raise
            return True
        else:
            logger.info(f'Using already downloaded {self.filepath}')
            return True

    def is_modified(self) -> bool:
        logger.info('Checking for file changes')
        s3_client = boto3.resource('s3')
        source = s3_client.meta.client.head_object(
            Bucket=self.bucket_name, Key=self.object_key)['LastModified'].strftime('%c')
        try:
            target = datetime.fromtimestamp(os.path.getmtime(
                self.filepath), tz=timezone.utc).strftime('%c')
        except FileNotFoundError:
            logger.info(f'File {self.filepath} doesnt exist.')
            return True
        if source == target:
            logger.info('Detected file changes')
            return False
        logger.info('No changes detected')
        return True

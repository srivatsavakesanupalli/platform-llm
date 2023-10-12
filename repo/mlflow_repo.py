import mlflow
from mlflow import MlflowClient
from loguru import logger
import os
import numpy as np
from typing import Dict, Optional
from exceptions.exceptions import EnvNotSet, InvalidModelLifestage, InvalidExperiment
from multiprocessing.pool import ThreadPool


class MLFRepo:
    '''MLFlow Repository operations'''

    def __init__(self, exp_id) -> None:
        self.lifestages = ['Production', 'Staging']
        self.experiment_name = exp_id
        try:
            self.uri = os.environ['MLFLOW_TRACKING_URI']
            mlflow.set_experiment(experiment_name=exp_id)
            mlflow.autolog(disable=True)
        except KeyError:
            raise EnvNotSet(
                'MLFLOW_TRACKING_URI is not set. Please configure this before starting the app')
        self.client = MlflowClient()

    def register_model(self, model_uri: str, model_name: str) -> bool:
        '''Register the model given by model_uri under the name given by model_name

        Args:
            model_uri: MLFlow run uri in the format runs://{runid}/{model_loc}
            model_name: Name of the model in the repository

        Returns:
            True if the registration is successful and False otherwise
        '''
        try:
            mlflow.register_model(model_uri, model_name)
            return True
        except Exception as e:
            logger.info(f'Model registration failed: {e}')
            return False

    def check_and_tag(self, check_only: bool = False) -> Optional[str]:
        '''Check the runs inside an experiment_name and tag them with version IDs if they dont have one
        Args:
            experiment_name: Experiment Name in mlflow that corresponds to an EXP_ID in CIBI
        '''
        exp_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        runs = mlflow.search_runs(exp_id)
        try:
            current_version = np.max(
                runs['tags.run_version'].fillna(-1).astype(int))
        except Exception as e:
            logger.info(
                f'Fetching version failed. Setting the current version to 1: \n {e}')
            current_version = 0
        if current_version == -1:
            current_version = 0
        if check_only:
            return current_version
        try:
            mlflow.end_run()
        except Exception as e:
            logger.info('No runs to end')

        def tag_run_version(run_id, version):
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag('run_version', version)
        # Look for tags.run_version column, create with current version + 1 if it doesnt exist. Else fill the empty values with current version + 1
        if 'tags.run_version' not in runs.columns:
            runs = runs['run_id'].to_list()
            args = [(run, current_version + 1) for run in runs]
        else:
            args = [(run['run_id'], current_version + 1)
                    for ix, run in runs.iterrows() if not run['tags.run_version']]
        with ThreadPool(os.cpu_count() - 1) as pool:
            pool.starmap(tag_run_version, args)
        logger.info(f'Current version set to {current_version + 1}')
        
    def get_best_model_runid(self) -> str:
        '''Get the run id corresponding to the best model in an experiment

        Args:
            experiment_name : MLFlow experiment name to search the best model in
            **kwargs can be used to pass a custom filteration criteria. Defaults to Pycaret filtration criteria
        Returns:
            A run id corresponding to the best tuned model in the experiment
        '''
        current_version = self.check_and_tag(check_only=True)
        exp_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        runs = mlflow.search_runs(exp_id)
        runs = runs.sort_values(by='start_time', ascending=False)
        if not current_version:
            raise InvalidExperiment(
                f'Experiment {self.experiment_name} is not properly configured')
        best_model_run_id = runs[(runs['tags.run_version'] == str(current_version))]["run_id"].tolist()[0]
        return best_model_run_id

    def push_artifact_folder(self, run_id: str, folder_name: str) -> bool:
        '''Push local model artifacts in a folder to MLFlow

        Args:
            run_id: run_id to which the artifacts need to be pushed
            folder_name: Path to the folder containing the artifacts

        Returns:
            True if the push is successful and False otherwise
        '''
        try:
            folder_key = folder_name.split(os.sep)[-1]
            mlflow.end_run()  # end any existing runs
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifacts(folder_name, folder_key)
            return True
        except Exception as e:
            logger.info(f'Pushing artifacts failed: {e}')
            return False

    def push_model(self, run_id: str, model) -> bool:
        '''Push local model artifacts in a folder to MLFlow

        Args:
            run_id: run_id to which the artifacts need to be pushed
            folder_name: Path to the folder containing the artifacts

        Returns:
            True if the push is successful and False otherwise
        '''
        try:

            mlflow.end_run()  # end any existing runs
            with mlflow.start_run(run_id=run_id):
                mlflow.pytorch.log_model(model, 'model')
        except Exception as e:
            logger.info(f'Pusing Model failed: {e}')
            return False

    def push_metric(self, run_id: str, metric: Dict, step: Optional[int]) -> bool:
        '''Push a metric to MLFlow

        Args:
            run_id: run_id to which the artifacts need to be pushed
            metric: A dictionary containing the name and value of the metric

        Returns:
            True if the push is successful and False otherwise
        '''
        try:
            mlflow.end_run()  # end any existing runs
            with mlflow.start_run(run_id=run_id):
                for metric_name in metric:
                    mlflow.log_metric(metric_name, metric[metric_name], step=step)
        except Exception as e:
            logger.info(f'Pusing Model failed: {e}')
            return False

    def runid_to_lifestage(self, runid: str, model_name: str, lifestage: str = 'Production') -> None:
        '''Changes the lifestage of the version of the model given by the unique model_name, corresponding the input runid. 
        Also changes the previously kept version in the same lifestage to Archived

        Args:
            runid : Unique id of the run, to which lifestage change is applicable
            model_name : Unique name of the model in the repository to change the lifestage
            lifestage : New lifestage of this model version

        '''
        if lifestage not in self.lifestages:
            raise InvalidModelLifestage(
                f'Model lifestage {lifestage} is not a valid one. It has to be one of {self.lifestages}')
        version = None
        old_version = None
        try:
            versions = self.client.search_model_versions(
                f"name='{model_name}'")
            for ver in versions:
                ver_dict = dict(ver)
                if ver_dict['run_id'] == runid:
                    version = ver_dict['version']
                if ver_dict['current_stage'] == lifestage:
                    old_version = ver_dict['version']
            if not version:
                logger.info(
                    f'Run ID {runid} does not have an attached model version')
                return
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=lifestage
            )
            logger.info(
                f'{model_name}:{version} corresponding to run {runid} has been set to {lifestage}')
            if not old_version:
                logger.info(
                    f'Lifestage {lifestage} was not previously set for this model. Nothing to Archive')
                return
            if old_version == version:
                logger.info(
                    f'Lifestage {lifestage} is already set to run {runid}')
                return
            self.client.transition_model_version_stage(
                name=model_name,
                version=old_version,
                stage="Archived"
            )
            logger.info(
                f'Previous {lifestage} model {model_name}:{old_version} has been archived')
        except Exception as e:
            logger.info(e)

    def exp_id_to_lifestage(self, model_name: str, lifestage: str, **kwargs) -> None:
        '''Changes the lifestage of the version of the model given by the unique model_name, corresponding the input experiment name. 
        Also changes the previously kept version in the same lifestage to Archived

        Args:
            experiment_name : experiment_name to which lifestage change is applicable
            model_name : Unique name of the model in the repository to change the lifestage
            lifestage : New lifestage of this model version
        '''
        best_run_id = self.get_best_model_runid(**kwargs)
        self.runid_to_lifestage(best_run_id, model_name, lifestage)

    def download_artifact(self, run_id, artifact_path):
        return mlflow.artifacts.download_artifacts(run_id=run_id,
                                    artifact_path=artifact_path)

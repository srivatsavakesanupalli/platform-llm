from base_infer import BaseInfer
# from torchcam.methods import SmoothGradCAMpp
from mlflow import MlflowClient
import os
import json
from PIL import Image
import time
import torch
import numpy as np
import cv2
import base64
from storage.factory import StorageFactory
from uuid import uuid4
from utils import unzip_file, get_embeddings, cosine_distance
from glob import glob
from loguru import logger
import mlflow
from datetime import datetime
import pandas as pd


class ClassificationInfer(BaseInfer):
    def __init__(self) -> None:
        super().__init__()
        logger.info(f"Loading model : models:/{self.model_name}/{self.model_version}")
        self.model = mlflow.pytorch.load_model(
            model_uri=f"models:/{self.model_name}/{self.model_version}").to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = True
        # self.cam_extractor = SmoothGradCAMpp(
        #     self.model, target_layer=self.model.layer4)
        self.run_id = self.get_run_id()
        artifact_path = "class_index"
        # Specify the local path to which to download the artifact
        local_dir = f"temp/{self.exp_id}"
        # Make sure the directory exists
        os.makedirs(local_dir, exist_ok=True)
        client = MlflowClient()
        client.download_artifacts(self.run_id, artifact_path, local_dir)
        self.labels = self.get_labels()
        self.train_emb = self.get_train_embs()

    def get_labels(self):
        json_file_path = f"temp/{self.exp_id}/class_index/class_index.json"
        with open(json_file_path, 'r') as f:
            labels_loaded = json.load(f)
        return labels_loaded

    def infer(self, image: Image):
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        input_batch = input_batch.to(self.device)
        self.model = self.model.to(self.device)

        output = self.model(input_batch)
        probabilities = torch.softmax(output[0], dim=0)
        predicted_class_index = torch.argmax(probabilities).item()
        prediction = predicted_class_index

        for k, v in self.labels.items():
            if str(v) == str(prediction):
                prediction = k
                break

        response = {
            'prediction': prediction,
            'confidence': probabilities[predicted_class_index].item(),
        }

        return response

    def infer_batch_images(self, images: torch.Tensor):
        images = images.to(self.device)
        self.model = self.model.to(self.device)

        output = self.model(images)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_indices = torch.argmax(probabilities, dim=1).tolist()

        responses = []
        for i in range(len(predicted_class_indices)):
            prediction = predicted_class_indices[i]
            for k, v in self.labels.items():
                if str(v) == str(prediction):
                    prediction = k
                    break

            response = {
                'prediction': prediction,
                'confidence': probabilities[i][predicted_class_indices[i]].item(),
            }
            responses.append(response)

        return responses

    def explain(self):
        return

    def infer_batch(self):
        start_time = time.time()
        batch_id = uuid4().hex[:8]
        data_loc = f"temp/mlflow/{self.exp_id}/"
        os.makedirs(data_loc, exist_ok=True)
        file_path = data_loc+"dataset.zip"
        storage = StorageFactory.create(
            file_path, self.batch_input)
        storage.download()
        unzip_file(zip_filepath=file_path, extract_location=data_loc)

        results = []
        temp_dirs = glob(data_loc+"*")
        temp_dir = [dirs for dirs in temp_dirs if (
            dirs.split("/")[-1][0] != "_") and (os.path.isdir(dirs))][0]

        all_images = [im for im in os.listdir(temp_dir) if (
            im.endswith(".jpg") or im.endswith(".png") or im.endswith(".jpeg"))]
        try:
            self.db['batch_status'].insert_one(
                {'exp_id': self.exp_id, 'start_time': start_time, 'batch_id': batch_id, 'batch_size': len(all_images), 'status': 'IN-PROGRESS'})
        except Exception as e:
            logger.info(f'Status update failed. {e}')

        if all_images:
            if len(all_images) % self.batch_size == 0:
                n_batches = len(all_images)//self.batch_size
            else:
                n_batches = len(all_images)//self.batch_size + 1
            embs = []
            for i in range(n_batches):
                images = all_images[i*self.batch_size: (i+1)*self.batch_size]
                images_path = [os.path.join(temp_dir, image)
                               for image in images]
                logger.info(images_path)
                loaded_images = torch.stack(
                    [
                        self.preprocess(Image.open(os.path.join(
                        temp_dir, image))) for image in images
                    ]
                )
                eval_embs = get_embeddings(
                    images_path, model=self.model).detach().cpu().numpy()
                embs.append(eval_embs)

                responses = self.infer_batch_images(loaded_images)
                for idx, response in enumerate(responses):
                    results.append(
                        [images[idx], response['prediction'], response['confidence']])

            eval_emb = np.vstack(embs).mean(axis=0)
            logger.info(self.train_emb.shape)
            logger.info(eval_emb.shape)
            drift_measure = cosine_distance(self.train_emb, eval_emb)
            with mlflow.start_run(run_id=self.run_id):
                mlflow.log_metric("drift_measure", len([i for i in drift_measure if i > 0.5])/len(drift_measure))

        # Convert the results to a pandas DataFrame and save as CSV
        logger.info("Batch inference Done!")

        df = pd.DataFrame(results, columns=[
            'image_name', 'prediction', 'confidence'])
        csv_path = os.path.join(temp_dir, 'predictions.csv')
        logger.info("saving batch csv")
        os.makedirs(f'predictions/{self.model_version}', exist_ok=True)
        df.to_csv(
            f'predictions/{self.model_version}/{datetime.now().strftime("%d%m%Y%H%M%S")}.csv', index=False)
        self.log_predictions()
        try:
            self.db['batch_status'].update_one(
                {'exp_id': self.exp_id, 'batch_id': batch_id}, {'$set': {'status': 'DONE', 'end_time': time.time()}})
        except Exception as e:
            logger.info(f'Status update failed. {e}')
        return csv_path

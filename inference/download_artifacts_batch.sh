#!/bin/bash
mlflow artifacts download -u models:/$MODEL_NAME/$STATE -d temp/mlflow
# pip install -r temp/mlflow/requirements.txt
python infer_batch.py
#!/bin/bash
mlflow artifacts download -u models:/$MODEL_NAME/$STATE -d temp/mlflow
#pip install -r temp/mlflow/requirements.txt
#gunicorn app:app --worker-connections 100  --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:80 --preload --timeout 300 --keep-alive 10 --graceful-timeout 120
uvicorn app:app --port 80 --host 0.0.0.0 --timeout-keep-alive 10 --timeout-graceful-shutdown 300 --limit-concurrency 10000
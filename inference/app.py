
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from starlette_exporter import PrometheusMiddleware, handle_metrics
import os
from PIL import Image
import io
from loguru import logger
from db import get_collections, validate_deploy_token
from typing import Annotated
from factory import InferFactory


db_data = get_collections()
DB = db_data['db']
PROJ = db_data['proj']
EXP = db_data['exp']

MODEL_NAME = os.environ["MODEL_NAME"]
EXP_ID = os.environ["EXP_ID"]

logger.info(f'Fetching configuration for {MODEL_NAME}')
data = EXP.find({'exp_id': EXP_ID})[0]
config = data['train_config']
logger.info(f'TRAIN CONFIG : \n {config}')

ENGINE = InferFactory.create(config['model']['type'])

app = FastAPI(root_path=f'/{MODEL_NAME}')

app.add_middleware(
    PrometheusMiddleware,
    app_name=f"{MODEL_NAME}-infer",
    group_paths=True, prefix=MODEL_NAME.replace('-', '_'),
    buckets=[0.1, 0.25, 0.5],
    filter_unhandled_paths=True,
    skip_paths=['/health', '/docs', '/metrics',
                '/', '/openapi.json', '/favicon.ico']
)
app.add_route("/metrics", handle_metrics)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
async def health():
    return {'status': "ALIVE"}


@app.post('/infer')
async def realtime_inference(deploy_token: Annotated[str, Header()], file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    if validate_deploy_token(deploy_token):
        return ENGINE.infer(image)
    else:
        raise HTTPException(status_code=403, detail="Invalid deploy token")


@app.post('/explain')
async def realtime_explain(deploy_token: Annotated[str, Header()], file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    if validate_deploy_token(deploy_token):
        return ENGINE.explain(image)
    else:
        raise HTTPException(status_code=403, detail="Invalid deploy token")


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=80)

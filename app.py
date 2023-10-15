from fastapi import FastAPI, Depends, Query
from configs.input_config import InputConfig
from configs.vendor_config import CloudVendor
from configs.loadtest_config import LoadTestConfig
from trainer.factory import TrainerFactory
from db import get_collections
from exceptions.exceptions import EnvNotSet, VendorNotConfigured
import time
import json
import os
from pathlib import Path
from loguru import logger
from typing_extensions import Annotated
from datetime import timedelta
from db import get_collections, get_deploy_token
from deployer.factory import DeployFactory
from pydantic import ValidationError
from configs.input_config import InputConfig
from configs.deployer_config import DeployerConfig
from configs.trainer_resource_config import TrainerResourceConfig
from typing import Dict, Literal, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import FastAPI, Form, HTTPException, Depends, status
from starlette_exporter import PrometheusMiddleware, handle_metrics
from utils import (
    generate_proj_id,
    generate_exp_id,
    get_ingress_model_name,
    update_experiment_state,
    get_run_status,
    get_latest_run_id,
    get_metrics,
    generate_preSignedURLs,
    list_s3_objects,
    mapPreSignedURLs,
)
from auth import (
    Token,
    authenticate_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_user,
    get_current_active_user,
    get_current_user,
    User,
    get_password_hash,
    UpdateUserData,
    generate_deployment_token,
    refresh_token,
)

try:
    VENDOR = os.environ["CLOUD_VENDOR"]
except KeyError:
    raise EnvNotSet("CLOUD_VENDOR env is not set")

try:
    CloudVendor[VENDOR]
except KeyError:
    raise VendorNotConfigured(f"{VENDOR} is not configured to use within this platform")

app = FastAPI(root_path=os.environ["ROOT_PATH"])
# app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    PrometheusMiddleware,
    app_name="llm-api",
    group_paths=True,
    prefix="llm",
    buckets=[0.1, 0.25, 0.5],
    filter_unhandled_paths=True,
    skip_paths=["/health", "/docs", "/metrics", "/", "/openapi.json", "/favicon.ico"],
)
app.add_route("/metrics", handle_metrics)

db_data = get_collections()
DB = db_data["db"]
PROJ = db_data["proj"]
EXP = db_data["exp"]
users_collection = DB["user_data"]
batch_collection = DB["batch_status"]
loadtest_collection = DB["loadtest"]


@app.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {
        "name": user.full_name,
        "access_token": access_token,
        "token_type": "bearer",
    }


@app.post("/extend_token")
async def extend_token(new_token: Annotated[str, Depends(refresh_token)]):
    return {"access_token": new_token, "token_type": "bearer"}


@app.post("/signup")
async def signup_user(username: str, password: str):
    user = get_user(username)
    if not user:
        hashed_password = get_password_hash(password)
        user_data = {"username": username, "hashed_password": hashed_password}
        users_collection.insert_one(user_data)
        return {"status": "SUCCESS"}
    else:
        return {"status": "FAILURE", "detail": f"User {username} already exists"}


@app.post("/update_user")
async def update_user(
    data: UpdateUserData, user: Annotated[User, Depends(get_current_user)]
):
    try:
        logger.info(user)
        user_data = list(users_collection.find({"username": user.username}))[0]
        user_data["disabled"] = False
        logger.info(user_data)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )
    users_collection.update_many(
        {"username": user.username}, {"$set": {**dict(data), **user_data}}
    )
    return {"status": "SUCCESS"}


@app.post("/create_project")
async def create_project(
    current_user: Annotated[User, Depends(get_current_active_user)],
    project_name: str = Form(...),
    description: str = Form(...),
):
    project_id = generate_proj_id()
    username = current_user.username
    project_data = {
        "proj_name": project_name,
        "proj_id": project_id,
        "proj_type": "llm",
        "description": description,
        "username": username,
    }
    PROJ.insert_one(
        {
            **project_data,
            "ts": int(time.time() * 1000),
            "created_ts": int(time.time() * 1000),
        }
    )
    return project_data

@app.post('/data_preview')
async def get_data_preview(current_user: Annotated[User, Depends(get_current_active_user)], input_path: str):
    # Read from a public dataset
    extn = input_path.split('.')[-1]
    # Support for azure SAS urls
    if '?' in extn:
        extn = extn[:extn.index('?')]
    if extn == 'csv':
        data = pd.read_csv(input_path)
    elif extn == 'parquet':
        data = pd.read_parquet(input_path)
    else:
        raise HTTPException(
            status_code=501, detail=f"File format {extn} is not supported")
    rows, columns = data.shape
    sample_data = data.head(100).to_json()
    response = {'sample_data': sample_data,
                'n_rows': rows, 'n_columns': columns}
    return response
    
@app.post("/create_exp")
async def create_exp(
    current_user: Annotated[User, Depends(get_current_active_user)],
    project_id: str = Form(...),
    experiment_name: str = Form(...),
):
    if not project_id:
        raise HTTPException(status_code=501, detail="Project ID cannot be empty")
    username = current_user.username
    try:
        _ = PROJ.find({"proj_id": project_id, "username": username})[0]
    except Exception:
        e = f"Project {project_id} does not belong to {username} or it doesnt exist"
        logger.info(e)
        raise HTTPException(status_code=403, detail=e)
    experiment_id = generate_exp_id(project_id)
    exp_data = {
        "exp_id": experiment_id,
        "exp_name": experiment_name,
        "proj_id": project_id,
        "username": username,
        "status": "CREATED",
    }
    EXP.insert_one({**exp_data, "ts": int(time.time() * 1000)})
    return exp_data


@app.post("/update_exp")
async def update_exp(
    current_user: Annotated[User, Depends(get_current_active_user)],
    exp_id: str = Form(...),
    status: Literal["CREATED", "PREPARED", "TRAINING", "TRAINED", "DEPLOYED"] = Form(
        ...
    ),
):
    username = current_user.username
    experiment = EXP.find_one({"exp_id": exp_id, "username": username})
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    update_experiment_state(
        exp_id=experiment["exp_id"], username=experiment["username"], state=status
    )
    return {"status": "SUCCESS"}


@app.post("/projects")
async def list_projects(
    current_user: Annotated[User, Depends(get_current_active_user)],
    proj_type: str = Query(None),
):
    username = current_user.username
    pipeline = [
        {"$match": {"username": username, "proj_type": proj_type}},
        {
            "$lookup": {
                "from": "experiments",
                "localField": "proj_id",
                "foreignField": "proj_id",
                "as": "exps",
            }
        },
        {
            "$project": {
                "proj_name": 1,
                "proj_id": 1,
                "proj_type": 1,
                "description": 1,
                "username": 1,
                "ts": 1,
                "experiments": {"$size": "$exps"},
            }
        },
    ]
    user_projects = list(PROJ.aggregate(pipeline))
    logger.info(user_projects)
    user_projects = [
        {k: v for k, v in item.items() if k != "_id"} for item in user_projects
    ]
    return user_projects


@app.post("/experiments")
async def list_experiments(
    current_user: Annotated[User, Depends(get_current_active_user)],
    project_id: Optional[str] = Form(None),
    experiment_id: Optional[str] = Form(None),
):
    username = current_user.username
    if project_id:
        user_experiments = list(
            EXP.find(
                {"proj_id": project_id, "username": username},
                {"exp_id": 1, "exp_name": 1, "status": 1, "ts": 1},
            )
        )
    elif experiment_id:
        user_experiments = list(
            EXP.find({"exp_id": experiment_id, "username": username})
        )
    else:
        raise HTTPException(
            status_code=500,
            detail="Please provide either a Project ID or an Experiment ID. Both cant be empty",
        )
    logger.info(user_experiments)
    user_experiments = [
        {k: v for k, v in item.items() if k != "_id"} for item in user_experiments
    ]
    return user_experiments


@app.delete("/experiments")
async def delete_experiments(
    current_user: Annotated[User, Depends(get_current_active_user)],
    experiment_id: str = Form(...),
):
    username = current_user.username
    if experiment_id:
        user_experiments = list(
            EXP.find({"exp_id": experiment_id, "username": username})
        )
        if len(user_experiments) > 0:
            try:
                EXP.delete_one({"exp_id": experiment_id, "username": username})
                return {"status": "SUCCESS"}
            except:
                raise HTTPException(
                    status_code=500, detail="No experiment found with the given exp_id"
                )
                return {"status": "FAILED"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Please provide an Experiment ID. Both cant be empty",
        )
        return {"status": "FAILED"}


@app.get("/pretrained_models")
async def list_pretrained(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    with open("pretrained_configs.json", "r") as f:
        config = json.load(f)

    return [{"name": model["name"], "link": model["link"]} for model in config]


@app.get("/finetuning_config")
async def get_finetuning_config(
    current_user: Annotated[User, Depends(get_current_active_user)], model_name: str
):
    with open("pretrained_configs.json", "r") as f:
        config = json.load(f)

    return [
        {"name": model["name"], "peft": model["peft"]}
        for model in config
        if model["name"] == model_name
    ]


@app.post("/deployment_token")
async def get_deployment_token(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    username = current_user.username
    token = generate_deployment_token()
    users_collection.update_many(
        {"username": username}, {"$set": {"deploy_token": token}}
    )
    return {"deploy_token": token}


@app.post("/deploy")
async def deploy_model(
    current_user: Annotated[User, Depends(get_current_active_user)],
    exp_id: str = Form(...),
):
    logger.info(f"Fetching deployment configuration for {exp_id}")
    username = current_user.username
    try:
        data = EXP.find({"exp_id": exp_id, "username": username})[0]
        config = data["deploy_config"]
        config["exp_id"] = exp_id
        config["proj_id"] = data["proj_id"]
        config["username"] = username
    except Exception:
        e = "Configuration not found"
        logger.info(e)
        raise HTTPException(detail=e)
    logger.info(f"DEPLOYER CONFIG : \n {config}")
    logger.info("Validating configuration")
    try:
        config = DeployerConfig(**config)
    except ValidationError:
        raise HTTPException(detail="Invalid Configuration found")
    logger.info("Initialising deployer")
    deployer = DeployFactory.create(config, backend="k8s")
    deployer.deploy()
    update_experiment_state(exp_id=exp_id, username=username, state="DEPLOYED")
    return {"status": "SUCCESS", "model_name": deployer.model_name}


@app.post("/train")
async def train_model(
    current_user: Annotated[User, Depends(get_current_active_user)],
    backend: Optional[Literal["cpu", "gpu"]] = "cpu",
    exp_id: str = Form(...),
):
    logger.info(f"Fetching configuration for {exp_id}")
    username = current_user.username
    try:
        data = EXP.find({"exp_id": exp_id, "username": username})[0]
        config = data["train_config"]
        proj_id = data["proj_id"]
    except Exception:
        e = "Configuration not found"
        logger.info(e)
        raise HTTPException(detail=e)

    logger.info("Creating Trainer Resource Configuration")
    config = {}
    config["exp_id"] = exp_id
    config["proj_id"] = proj_id
    config["username"] = username
    config["backend"] = backend
    config = TrainerResourceConfig(**config)
    logger.info(f"TRAIN REOURCES CONFIG : \n {config}")
    logger.info("Initialising trainer deployment")
    trainer_deployer = DeployFactory.create(config, backend="train")
    trainer_deployer.deploy()
    return {"status": "SUCCESS", "model_name": trainer_deployer.model_name}


@app.post("/training_status")
async def get_training_status(
    current_user: Annotated[User, Depends(get_current_active_user)],
    exp_id: str = Form(...),
):
    username = current_user.username
    status = {}
    try:
        data = EXP.find({"exp_id": exp_id, "username": username})[0]
        status["exp_status"] = data["status"] if "status" in data else "FAILED"
        status["from"] = data["ts"] if "ts" in data else ""
        status["to"] = data["to"] if "to" in data else int(time.time() * 1000)
        status["scatter_plot"] = data["tSNE"] if "tSNE" in data else ""
        status["train_loss"] = {}
        status["val_loss"] = {}
    except Exception as e:
        logger.info(e)
        raise HTTPException(
            status_code=400,
            detail=f"{current_user} doesnt have access to {exp_id} experiment",
        )
    return status


@app.get("/batch_status")
async def get_batch_status(
    current_user: Annotated[User, Depends(get_current_active_user)], exp_id: str
):
    try:
        data = list(
            batch_collection.find({"exp_id": exp_id}, {"_id": 0}).sort("ts", -1)
        )
        n_runs = len(data)
        data = data[:10]
        output = {"status": "SUCCESS", "n_runs": n_runs, "latest_runs": data}
    except Exception as e:
        logger.info(f"Error fetching batch runs for exp {exp_id}: {e}")
        output = {"status": "FAILURE"}
    return output


@app.post("/loadtest")
async def deploy_loadtest(
    current_user: Annotated[User, Depends(get_current_active_user)],
    exp_id: str = Form(...),
):
    logger.info(f"Fetching load testing configuration for {exp_id}")
    username = current_user.username
    try:
        data = EXP.find({"exp_id": exp_id, "username": username})[0]
        config = data["loadtest_config"]
        config["exp_id"] = exp_id
        config["proj_id"] = data["proj_id"]
        ingress = get_ingress_model_name(data["proj_id"].lower())
        ingress = f"http://{ingress}"
        config["ingress"] = ingress

    except Exception:
        e = "Configuration not found"
        logger.info(e)
        raise HTTPException(detail=e)
    deploy_token = get_deploy_token(username)["deploy_token"]
    config["deploy_token"] = deploy_token
    logger.info(f"LOADTEST CONFIG : \n {config}")
    logger.info("Validating configuration")
    try:
        config = LoadTestConfig(**config)
    except ValidationError:
        raise HTTPException(detail="Invalid Configuration found")
    logger.info("Initialising deployer")
    deployer = DeployFactory.create(config, backend="locust")
    deployer.deploy()
    return {"status": "SUCCESS", "model_name": deployer.model_name}


@app.get("/loadtest_status")
async def get_batch_status(
    current_user: Annotated[User, Depends(get_current_active_user)], exp_id: str
):
    try:
        data = list(loadtest_collection.find({"exp_id": exp_id}).sort("ts", -1))[0]
        output = {"status": data["status"]}
    except Exception as e:
        logger.info(f"Error fetching batch runs for exp {exp_id}: {e}")
        output = {"status": "NOTSTARTED"}
    return output


@app.post("/create_dataset")
async def create_dataset_from_annotations(
    current_user: Annotated[User, Depends(get_current_active_user)],
    problem_type: Literal["classification", "detection", "segmentation"],
    annotations: Dict,
):
    logger.info(annotations)
    return


@app.post("/update_config")
async def update_config(
    current_user: Annotated[User, Depends(get_current_active_user)],
    exp_id: str,
    config_type: Literal["train", "deploy", "loadtest"],
    config: Dict,
):
    username = current_user.username

    # Verify if the experiment exists
    experiment = EXP.find_one({"exp_id": exp_id, "username": username})
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    key = f"{config_type}_config"

    # Update the experiment with the new configurations
    EXP.update_one(
        {"exp_id": exp_id, "username": username},
        {
            "$set": {
                key: config,
            }
        },
    )

    return {"status": "SUCCESS"}


@app.post("/ingress")
async def get_ingress(
    current_user: Annotated[User, Depends(get_current_active_user)], project_id: str
):
    model_name = project_id.lower()
    ingress = get_ingress_model_name(model_name)
    return {"ingress": ingress}


@app.post("/replica_suggestion")
async def get_min_replicas(
    current_user: Annotated[User, Depends(get_current_active_user)],
    exp_id: str,
    min_concurrency: int,
):
    try:
        loadtest_results = list(DB["loadtest"].find({"exp_id": exp_id}).sort("ts", -1))[
            0
        ]
        fail_users = int(loadtest_results["fail_users"] * 0.8)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=404, detail="No loadtest found")
    if min_concurrency > fail_users:
        min_replicas = min_concurrency // fail_users + 1
    else:
        min_replicas = 1
    return {"min_replicas": min_replicas, "max_replicas": min_replicas + 2}


@app.post("/sample_request")
async def create_sample_request(
    current_user: Annotated[User, Depends(get_current_active_user)],
    exp_id: str = Form(...),
):
    model_name = ("-".join(exp_id.split("-")[0:2])).lower()
    curl = f"curl -X 'POST' '20.204.231.120/{model_name}/infer' -F 'image=@/path/to/image.png' "
    return {"request": curl}


@app.get("/cmpr")
async def get_cm_pr_values(
    current_user: Annotated[User, Depends(get_current_active_user)], exp_id: str
):
    try:
        data = list(
            EXP.find({"exp_id": exp_id, "username": current_user.username}).sort(
                "ts", -1
            )
        )[0]
        if data["train_config"]["model"]["type"] == "classification":
            output = data["performance_data"]["cm_pr"]
        else:
            output = []
    except Exception as e:
        logger.info(f"Error fetching batch runs for exp {exp_id}: {e}")
        output = []
    return output


@app.get("/model_metrics")
async def get_model_metrics(
    current_user: Annotated[User, Depends(get_current_active_user)], exp_id: str
):
    try:
        run_id = get_latest_run_id(exp_id)
        output = {
            "Accuracy": get_metrics(run_id, "Accuracy")["0"],
            "F1-Score": get_metrics(run_id, "F1-Score")["0"],
            "Precision": get_metrics(run_id, "Precision")["0"],
            "Recall": get_metrics(run_id, "Recall")["0"],
        }
    except Exception as e:
        logger.info(f"Error fetching batch runs for exp {exp_id}: {e}")
        output = {}
    return output


@app.get("/drift_metrics")
async def get_drift_metrics(
    current_user: Annotated[User, Depends(get_current_active_user)], exp_id: str
):
    try:
        run_id = get_latest_run_id(exp_id)
        drift = get_metrics(run_id, "drift_measure")
    except Exception as e:
        logger.info(f"Error fetching drift measure for exp {exp_id}: {e}")
        drift = {}
    return drift


@app.get("/get_misclassifications")
async def get_misclassified_files(
    current_user: Annotated[User, Depends(get_current_active_user)],
    exp_id: str,
    threshold: float,
):
    try:
        run_id = get_latest_run_id(exp_id)
        validation_files = list_s3_objects(exp_id, run_id)
        preSignedURLs = generate_preSignedURLs(validation_files)
        return mapPreSignedURLs(
            validation_files, preSignedURLs, str(threshold), exp_id, run_id
        )
    except Exception as e:
        logger.info(e)
        logger.info(f"Error fetching misclassifications for {exp_id}: {e}")
        return {}

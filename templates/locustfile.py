from locust import HttpUser, task
import logging
import gevent
import time
from locust import events
from locust.runners import STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP, MasterRunner, LocalRunner
from pymongo import MongoClient
import pymongo
import time
import os
import urllib.request
import requests

MONGO_URL = "MONGO_URL_VALUE"
MONGO_DB = "MONGO_DB_VALUE"
EXP_ID = "EXP_ID_VALUE"
DEPLOY_TOKEN = "DEPLOY_TOKEN_VALUE"

FAIL_RATIO = FAIL_RATIO_VALUE
AVG_RESPONSE_TIME = AVG_RESPONSE_TIME_VALUE
P90_RESPONSE_TIME = P90_RESPONSE_TIME_VALUE


def get_collections(uri: str = MONGO_URL):
    mongo_conf = {}
    uri_data = pymongo.uri_parser.parse_uri(uri)
    hostslist = [f'{x[0]}:{x[1]}' for x in uri_data['nodelist']]
    host, port = hostslist[0].split(':')
    mongo_conf = {
        **uri_data['options'],
        'host': f'mongodb://{host}:{port}',
        'username': uri_data['username'],
        'password': uri_data['password'],
    }
    conn = MongoClient(**mongo_conf)
    db = conn[MONGO_DB]
    collection = db['loadtest']
    return collection


class User(HttpUser):
    @task
    def compare_faces(self) -> None:
        self.client.post("/infer", files=[('file', ('sample.jpg', open('samples/sample.jpg', 'rb'), 'image/jpeg'))], headers={
            'Content-Type': 'application/json',
            'Deploy-Token': DEPLOY_TOKEN
        })


@events.test_start.add_listener
def download_image(environment, **kwargs):
    os.makedirs('samples', exist_ok=True)
    urllib.request.urlretrieve(
        'https://inspirationseek.com/wp-content/uploads/2016/02/Cute-Dog-Photography.jpg', filename='samples/sample.jpg')


@events.quitting.add_listener
def _(environment, **kw):
    if environment.stats.total.fail_ratio > FAIL_RATIO:
        logging.error(f"Test failed due to failure ratio > {FAIL_RATIO}%")
        users_requests(environment)
        environment.process_exit_code = 1
    elif environment.stats.total.avg_response_time > AVG_RESPONSE_TIME:
        users_requests(environment)
        logging.error(
            f"Test failed due to average response time ratio > {AVG_RESPONSE_TIME} ms")
        environment.process_exit_code = 1
    elif environment.stats.total.get_response_time_percentile(0.90) > P90_RESPONSE_TIME:
        users_requests(environment)
        logging.error(
            f"Test failed due to 90th percentile response time > {P90_RESPONSE_TIME} ms")
        environment.process_exit_code = 1
    else:
        environment.process_exit_code = 0


def users_requests(environment):
    env = environment
    current_users = env.runner.user_count
    current_requests = env.runner.stats.num_requests
    logging.info(str(env.runner))
    logging.info(f"no of requests: {str(current_requests)}")
    logging.info(f"current users: {str(current_users)}")
    collection = get_collections()
    collection.find_one_and_replace({'exp_id': EXP_ID}, {
                                    'exp_id': EXP_ID, 'fail_requests': current_requests, 'fail_users': current_users, 'ts': int(time.time()*1000)}, upsert=True)

    if isinstance(environment.runner, MasterRunner):
        try:
            requests.get('http://localhost:9646/quitquitquit')
        except Exception as e:
            logging.error(f'Auto shutdown failed : {e}')

    return current_users, current_requests


def checker(environment):
    while environment.runner.state not in [STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP]:
        time.sleep(1)
        if environment.runner.stats.total.fail_ratio > FAIL_RATIO:
            logging.error(f"Test failed due to failure ratio > {FAIL_RATIO}%")
            users_requests(environment)
            environment.runner.quit()
            return
        elif environment.runner.stats.total.avg_response_time > AVG_RESPONSE_TIME:
            logging.error(
                f"Test failed due to average response time ratio > {AVG_RESPONSE_TIME} ms")
            users_requests(environment)
            environment.runner.quit()
            return
        elif environment.runner.stats.total.get_response_time_percentile(0.90) > P90_RESPONSE_TIME:
            logging.error(
                f"Test failed due to 90th percentile response time > {P90_RESPONSE_TIME} ms")
            users_requests(environment)
            environment.runner.quit()
            return


@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    # dont run this on workers, we only care about the aggregated numbers
    if isinstance(environment.runner, MasterRunner) or isinstance(environment.runner, LocalRunner):
        gevent.spawn(checker, environment)

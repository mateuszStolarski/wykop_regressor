from bs4 import BeautifulSoup
import requests
import pandas as pd
from helpers import *
from celery import Celery
from celery.schedules import crontab
import consts
from movies import Movies
from extractor import *
import pymongo
from prometheus_metrics import set_metrics, proccessing_time, registry
from prometheus_client import push_to_gateway

BROKER_URL = consts.broker_url
REDIS_URL = consts.redis_url
WYKOP_URL = consts.wykop_url
REQUEST_TIME = proccessing_time
REGISTRY = registry

app = Celery('tasks', broker=BROKER_URL)


@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(crontab(minute='*/1'),
                             execute_pipeline.s(),
                             name='wykop by 1 minute'
                             )


@app.task
@REQUEST_TIME.time()
def execute_pipeline():
    chain = download_data.s() | save_df.s()
    chain()
    push_to_gateway(consts.gateway, job='batchA', registry=REGISTRY)


@app.task
def download_data():
    with requests.Session() as session:
        with session.get(
            url=WYKOP_URL,
            timeout=5
        ) as response:
            content = response.content

    soup = BeautifulSoup(content, features="html.parser")
    ul = soup.find("ul", {"id": consts.list_id})
    firstli = ul.find('li')

    columns = Movies.get_columns()
    results = pd.DataFrame(columns=columns)

    results = check_and_append(results, firstli, columns)
    for li in firstli.findNextSiblings():
        results = check_and_append(results, li, columns)

    results = send_data(results)

    return(results)


@app.task
def set_metrics(df):
    df = read_data(df)
    set_metrics(df)
    df = send_data(df)

    return(df)


@app.task
def save_df(df):
    client = pymongo.MongoClient(host=CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    db = client[consts.database_name]
    df = read_data(df)

    documents = []
    for _, row in df.iterrows():
        content = row.content.replace('"', '')
        content = content.replace('„', '')
        content = content.replace('”', '')

        document = Movies(
            author=str(row.author),
            creation_date=int(row.creation_date),
            content=str(content),
            number_of_pluses=int(row.number_of_pluses),
            number_of_comments=int(row.number_of_comments),
        )

        documents.append(document.__dict__)

    if len(documents) > 0:
        db[consts.collection_name].insert_many(documents=documents)

    return 'Ok'

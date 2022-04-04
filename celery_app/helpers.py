import json
import pandas as pd
import consts
import pymongo

CONNECTION_STRING = consts.mongo_connection_string
COLLECTION_NAME = consts.collection_name
DATABASE_NAME = consts.database_name


def send_data(df):
    df = df.to_json(orient="index")
    df = json.loads(df)
    df = json.dumps(df, indent=4)

    return df


def read_data(df):
    df = json.loads(df)
    df = pd.DataFrame.from_dict(df, orient='index')

    return df


def get_collection():
    client = pymongo.MongoClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    return collection


def alreadyExists(creation_date):
    collection = get_collection()

    if collection.count_documents({'creation_date': int(creation_date)}, limit=1):
        return True
    else:
        return False

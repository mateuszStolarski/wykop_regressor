from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from torch_model.loader import get_transformer, prepare_mlp_model, get_encoder
from torch_model.executor import model_prediction, get_device
from torch_model.persistance.context import session_factory
from torch_model.persistance.migrator import init_script
from torch_model.persistance.repository import is_empty
from torch_model.persistance.models.basics import *
from torch_model.persistance.models.ratings import *
from dto.request.movie import Movie
import pandas as pd
import numpy as np


class Loaded_Models:
    def __init__(self, session):
        self.tokenizer, self.transformer = get_transformer()
        self.mlp = prepare_mlp_model(session=session)
        self.encoder = get_encoder()
        self.device = get_device()


def register_cors(app: FastAPI):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def create_app() -> FastAPI:
    app = FastAPI(debug=False)
    session = session_factory()
    init_script(session=session)
    loaded_models = Loaded_Models(session)
    register_cors(app)

    return app, session, loaded_models


app, session, loaded_models = create_app()


@app.get("/", status_code=status.HTTP_200_OK)
def read_root():
    return {"Hello": "World"}


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(movie: Movie):
    df = pd.DataFrame(np.array([[movie.titleType,
                               movie.originalTitle,
                               movie.isAdult,
                               movie.startYear,
                               movie.runtimeMinutes,
                               movie.genres,
                               movie.numVotes]]),
                      columns=['titleType',
                               'originalTitle',
                               'isAdult',
                               'startYear',
                               'runtimeMinutes',
                               'genres',
                               'numVotes'])
    result = model_prediction(df=df,
                              model=loaded_models.mlp,
                              tokenizer=loaded_models.tokenizer,
                              transformer=loaded_models.transformer,
                              encoder=loaded_models.encoder)

    return {"Prediction": result}

from pandas import DataFrame
from .persistance.repository import *
from .executor import fit, get_device, to_dataloader, get_writer_windows
from .mlp_model import MLP_Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import torch.optim as optim
import torch
import pickle
import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Tuple
from sqlalchemy.orm import Session

STARTYEAR = 1895
BATCH_SIZE = 32
NUM_FEATURES = 128
NUM_EPOCHS = 20
LR = 1e-4
DIRECTORY = '.\\torch_model\\models'
MLP_FILE_NAME = 'MLP_model.sav'
TRANSFORMER_FILE_NAME = 'bert_model.sav'
TOKENIZER_FILE_NAME = 'bert_tokenizer.sav'
ENCODER_FILE_NAME = 'encoder.sav'
y_label = ['averageRating']


def get_transformer():
    if os.path.isdir(DIRECTORY) == False:
        os.makedirs(DIRECTORY, exist_ok=True)

    if os.path.isfile(f'{DIRECTORY}\\{TRANSFORMER_FILE_NAME}') != True or os.path.isfile(f'{DIRECTORY}\\{TOKENIZER_FILE_NAME}') != True:
        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

        pickle.dump(tokenizer, open(f'{DIRECTORY}\\{TOKENIZER_FILE_NAME}', 'wb'))
        pickle.dump(model, open(f'{DIRECTORY}\\{TRANSFORMER_FILE_NAME}', 'wb'))

    tokenizer = pickle.load(open(f'{DIRECTORY}\\{TOKENIZER_FILE_NAME}', 'rb'))
    model = pickle.load(open(f'{DIRECTORY}\\{TRANSFORMER_FILE_NAME}', 'rb'))

    return tokenizer, model


def get_encoder():
    return pickle.load(open(f'{DIRECTORY}\\{ENCODER_FILE_NAME}', 'rb'))


def preprocessed(data: DataFrame) -> Tuple[DataFrame, int]:
    tokenizer, model = get_transformer()

    if os.path.isdir(DIRECTORY) == False:
        os.makedirs(DIRECTORY, exist_ok=True)

    if os.path.isfile(f'{DIRECTORY}\\{ENCODER_FILE_NAME}') == False:
        enc = OrdinalEncoder()
        enc.fit(data[['titleType', 'genres', 'runtimeMinutes']])
        pickle.dump(enc, open(f'{DIRECTORY}\\{ENCODER_FILE_NAME}', 'wb'))

    enc = pickle.load(open(f'{DIRECTORY}\\{ENCODER_FILE_NAME}', 'rb'))
    data[['titleType', 'genres', 'runtimeMinutes']] = enc.transform(data[['titleType', 'genres', 'runtimeMinutes']])

    embedings = []
    for i in range(len(data)):
        content = str(data.originalTitle[i]),
        inputs = tokenizer(content[0], return_tensors="pt")
        outputs = model(**inputs)
        result = np.asarray(outputs.last_hidden_state.tolist()).flatten()
        embedings.append(result[0:NUM_FEATURES])

    columns = []
    for i in range(NUM_FEATURES):
        columns.append(i)
    data[columns] = embedings

    data = data.drop(columns=['originalTitle'])
    return data, len(data.columns) - 2


def save_test(X_test: list, y_test: list):
    np.savetxt("..\\data\\features.csv", X_test, delimiter=" ",  fmt='% s')
    np.savetxt("..\\data\\labels.csv", y_test, delimiter=" ",  fmt='% s')


def load_training_data_to_dataloader(session: Session, device='cpu', test_size: float = 0.33, random_state: int = 42):
    data = get_data_with_start_year(STARTYEAR, session)
    data, num_features = preprocessed(data)
    y = data[y_label]
    data = data.drop(columns=y_label)
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=test_size, random_state=random_state)

    save_test(X_test=X_test, y_test=y_test)

    X_train = X_train[X_train.columns[1:len(X_train.columns)]].to_numpy()
    X_test = X_test[X_test.columns[1:len(X_test.columns)]].to_numpy()

    X_train = torch.tensor(np.vstack(X_train).astype(np.float32)).to(device=device)
    y_train = torch.tensor(np.vstack(y_train.to_numpy()).astype(np.float32)).to(device=device).T
    X_test = torch.tensor(np.vstack(X_test).astype(np.float32)).to(device=device)
    y_test = torch.tensor(np.vstack(y_test.to_numpy()).astype(np.float32)).to(device=device).T

    train_dl, test_dl = to_dataloader(X_train, y_train[0].type(
        torch.LongTensor), X_test, y_test[0].type(torch.LongTensor), BATCH_SIZE)

    return train_dl, test_dl, num_features


def prepare_mlp_model(session: Session) -> torch.nn.Module:
    device = get_device()
    writer, _ = get_writer_windows()

    if os.path.isdir(DIRECTORY) == False:
        os.makedirs(DIRECTORY, exist_ok=True)

    if os.path.isfile(f'{DIRECTORY}\\{MLP_FILE_NAME}') == False:
        train_dl, test_dl, num_features = load_training_data_to_dataloader(session)
        model = MLP_Model(features=num_features)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_fn = torch.nn.L1Loss()

        fit(model, optimiser=optimizer, loss_fn=loss_fn, train_dl=train_dl,
            val_dl=test_dl, epochs=NUM_EPOCHS, writer=writer, device=device)

        pickle.dump(model, open(f'{DIRECTORY}\\{MLP_FILE_NAME}', 'wb'))

    model = pickle.load(open(f'{DIRECTORY}\\{MLP_FILE_NAME}', 'rb'))
    return model

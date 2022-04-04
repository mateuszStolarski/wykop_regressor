from pandas import DataFrame
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
import datetime
from tqdm import tqdm
import torch.optim as optim
import os
import numpy as np


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def to_dataloader(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    test_ds = TensorDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size, pin_memory=True)

    return train_dl, test_dl


def get_writer_windows():
    log_dir = f'..\\data\\logs\\{datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")}'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer, log_dir


def count_correct(y_pred: torch.Tensor, y_true: torch.Tensor
                  ) -> torch.Tensor:
    preds = torch.argmax(y_pred, dim=1)
    return (preds == y_true).float().sum()


def validate(
    model,
    loss_fn: torch.nn.L1Loss,
    dataloader: DataLoader,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    loss = 0
    correct = 0
    all = 0
    for X_batch, y_batch in dataloader:
        y_pred = model(X_batch.to(device))
        all += len(y_pred)
        loss += loss_fn(y_pred, y_batch.to(device)).sum()
        correct += count_correct(y_pred, y_batch.to(device))

    return loss / all


def fit(
    model: torch.nn.Module,
    optimiser: optim.Optimizer,
    loss_fn: torch.nn.CrossEntropyLoss,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    print_metrics: str = True,
    writer: torch.utils.tensorboard.writer.SummaryWriter = None,
    device: torch.device = torch.device('cpu')
):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in tqdm(train_dl):
            y_pred = model(X_batch.to(device))
            loss = loss_fn(y_pred, y_batch.to(device))

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        if print_metrics:
            model.eval()
            with torch.no_grad():
                train_loss = validate(model,
                                      loss_fn=loss_fn,
                                      dataloader=train_dl
                                      )
                val_loss = validate(model,
                                    loss_fn=loss_fn,
                                    dataloader=val_dl
                                    )

                print(
                    f"Epoch {epoch}: "
                    f"train loss = {train_loss:.3f} "
                    f"validation loss = {val_loss:.3f} "
                )

                if writer != None:
                    writer.add_scalars(
                        main_tag='loss',
                        tag_scalar_dict={
                            'train': train_loss,
                            'dev': val_loss
                        },
                        global_step=epoch+1
                    )


def model_prediction(df: DataFrame, model: torch.nn.Module, tokenizer, transformer, encoder, device='cpu'):
    embedings = []
    df[['titleType', 'genres', 'runtimeMinutes']] = encoder.transform(df[['titleType', 'genres', 'runtimeMinutes']])

    for _, row in df.iterrows():
        inputs = tokenizer(row.originalTitle, return_tensors="pt")
        outputs = transformer(**inputs)
        result = np.asarray(outputs.last_hidden_state.tolist()).flatten()
        embedings.append(result[0:128])
        row.isAdult = bool(row.isAdult)

    columns = []
    for i in range(128):
        columns.append(i)
    df = df.drop(columns=['originalTitle'])
    df = df[df.columns[1:len(df.columns)]]
    df = df.to_numpy()
    df = np.concatenate((df, embedings), axis=1)

    tensor = torch.tensor(np.vstack(df).astype(np.float32)).to(device=device)
    model.eval()
    return round(model(tensor).item(), 1)

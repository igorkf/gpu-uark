import argparse
import sys
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from sklearn import impute

from preprocessing import create_field_location
from nn_utils import DNA_Net, mean_mse, mean_r    


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=False)
parser.add_argument("--loss", type=str, required=False)
parser.add_argument("--bs", type=int, required=False)
parser.add_argument("--gamma", type=float, required=False)
parser.add_argument("--fl", type=int, required=False)
parser.add_argument("--device", type=str, required=False)
args = parser.parse_args()

VAL_INIT = 2022
if not (len(sys.argv) > 1):
    MODEL = "G" # only genetic data
    LOSS = "mse"
    BATCH_SIZE = 32
    GAMMA = 0.1
    FILTER_LOC = 0
    DEVICE = "cpu"
else:
    MODEL = args.model
    LOSS = args.loss
    BATCH_SIZE = args.bs
    GAMMA = args.gamma
    FILTER_LOC = args.fl
    DEVICE = args.device

EPOCHS = 100 # 500
ENV_WEIGHT = 0.1
PATIENCE = int(EPOCHS // 10)
SEED = 42

# set seed for reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

if DEVICE not in ["cpu", "cuda"]:
    raise ValueError("Invalid device. Choose 'cpu' or 'cuda'.")
device = torch.device(DEVICE)


def set_scheduler(optimizer, gamma):
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return scheduler


def train(model, optimizer, scheduler, train_loader, val_loader, epochs, patience):
    best_epoch = 0
    best_val = -np.inf
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_train_y_true = []
        all_train_y_pred = []
        all_train_envs = []
        for x_batch, y_batch, env_batch in train_loader:
            x_batch, y_batch, env_batch = x_batch.to(device), y_batch.to(device), env_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            if LOSS == "mse":
                loss_within = mean_mse(y_batch, outputs, env_batch)
                loss_env = mse(y_batch, outputs)  # global mse
                loss = (1 - ENV_WEIGHT) * loss_within + loss_env * ENV_WEIGHT
            elif LOSS == "r":
                raise Exception("Loss 'r' not implemented.")
                loss_within = -mean_r(y_batch, outputs, env_batch)
                loss = loss_within
                # loss_env = -cos(
                #     y_batch - y_batch.mean(dim=1, keepdim=True),
                #     outputs - outputs.mean(dim=1, keepdim=True),
                # )  # global r
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            all_train_y_true.append(y_batch)
            all_train_y_pred.append(outputs)
            all_train_envs.append(env_batch)
        train_loss /= len(train_loader)
        train_mean_r = mean_r(
            torch.cat(all_train_y_true),
            torch.cat(all_train_y_pred),
            torch.cat(all_train_envs),
        ).item()

        # Validation
        val_loss = 0
        all_val_y_true = []
        all_val_y_pred = []
        all_val_envs = []
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch, env_batch in val_loader:
                x_batch, y_batch, env_batch = x_batch.to(device), y_batch.to(device), env_batch.to(device)
                outputs = model(x_batch)
                if LOSS == "mse":
                    loss_within = mean_mse(y_batch, outputs, env_batch)
                    loss_env = mse(y_batch, outputs)  # global mse
                    loss = (1 - ENV_WEIGHT) * loss_within + loss_env * ENV_WEIGHT
                elif LOSS == "r":
                    raise Exception("Loss 'r' not implemented.")
                    loss_within = -mean_r(y_batch, outputs, env_batch)
                    loss = loss_within
                    # loss_env = -cos(
                    #     y_batch - y_batch.mean(dim=1, keepdim=True),
                    #     outputs - outputs.mean(dim=1, keepdim=True),
                    # )  # global r
                val_loss += loss.item()
                all_val_y_true.append(y_batch)
                all_val_y_pred.append(outputs)
                all_val_envs.append(env_batch)
        val_loss /= len(val_loader)
        val_mean_r = mean_r(
            torch.cat(all_val_y_true),
            torch.cat(all_val_y_pred),
            torch.cat(all_val_envs),
        ).item()

        if val_mean_r > best_val:
            best_val = val_mean_r
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                f"output/best_model_{MODEL}_{LOSS}_{BATCH_SIZE}_{GAMMA}_fl_{FILTER_LOC}_device_{DEVICE}.pt",
            )
            print(f"[{best_epoch + 1}] New best:", best_val)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

        print("Epoch:", epoch + 1, "/", EPOCHS)
        print("Training Loss:", train_loss)
        print("Validation Loss:", val_loss)
        print("Training Mean R:", train_mean_r)
        print("Validation Mean R:", val_mean_r)
        print(f"[{best_epoch + 1}] Best val:", best_val)
        print(f"Learning Rate: {scheduler.get_last_lr()[0]}")
        print()

        scheduler.step(val_mean_r)


def predict(model, x_tensor):
    x_tensor = x_tensor.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(x_tensor).cpu().numpy().ravel().tolist()
    return predictions


if __name__ == "__main__":

    # load
    xtrain = pd.read_csv(f"output/xtrain_fl_{FILTER_LOC}.csv")
    ytrain = pd.read_csv(f"output/ytrain_fl_{FILTER_LOC}.csv")
    xval = pd.read_csv(f"output/xval_fl_{FILTER_LOC}.csv")
    yval = pd.read_csv(f"output/yval_fl_{FILTER_LOC}.csv")

    xtrain = create_field_location(xtrain)
    # it has some values that are variations of a field location
    xtrain["Field_Location"] = xtrain["Field_Location"].str[:4].astype("category")
    xval = create_field_location(xval)
    xval["Field_Location"] = xval["Field_Location"].astype("category")

    # get env groups
    xtrain_envs = xtrain["Env"].values.tolist()
    xval_envs = xval["Env"].values.tolist()
    envs = xtrain_envs + xval_envs
    print("# unique envs:", len(set(envs)))
    le = preprocessing.LabelEncoder()
    envs_int = le.fit_transform(envs).tolist()
    envs2int = {x: i for x, i in zip(envs, envs_int)}
    int2envs = {i: x for x, i in zip(envs, envs_int)}
    xtrain_envs = [envs2int[x] for x in xtrain_envs]
    xval_envs = [envs2int[x] for x in xval_envs]

    # get loc groups
    xtrain_locs = xtrain["Field_Location"].values.tolist()
    xval_locs = xval["Field_Location"].values.tolist()
    locs = xtrain_locs + xval_locs
    print("# unique locs:", len(set(locs)))
    le = preprocessing.LabelEncoder()
    locs_int = le.fit_transform(locs).tolist()
    locs2int = {x: i for x, i in zip(locs, locs_int)}
    locs2envs = {i: x for x, i in zip(locs, locs_int)}
    xtrain_locs = [locs2int[x] for x in xtrain_locs]
    xval_locs = [locs2int[x] for x in xval_locs]

    # set index
    xtrain = xtrain.set_index(["Env", "Hybrid"]).drop("Field_Location", axis=1)
    xval = xval.set_index(["Env", "Hybrid"]).drop("Field_Location", axis=1)
    ytrain = ytrain.set_index(["Env", "Hybrid"])
    yval = yval.set_index(["Env", "Hybrid"])

    # Extract DNA information
    imputer = impute.SimpleImputer(strategy="mean")
    scaler = preprocessing.StandardScaler()
    xtrain_dna = xtrain.iloc[:, 81:].values
    xval_dna = xval.iloc[:, 81:].values
    xtrain_dna = imputer.fit_transform(xtrain_dna)
    xval_dna = imputer.transform(xval_dna)
    xtrain_dna = scaler.fit_transform(xtrain_dna)
    xval_dna = scaler.transform(xval_dna)

    # Extract Environmental information
    # scaler = preprocessing.StandardScaler()
    # xtrain_environ = xtrain.iloc[:, :81].values
    # xval_environ = xval.iloc[:, :81].values
    # xtrain_environ = scaler.fit_transform(xtrain_environ)
    # xval_environ = scaler.transform(xval_environ)

    # Convert data to PyTorch tensors
    xtrain_dna_tensor = torch.tensor(xtrain_dna, dtype=torch.float32)
    # xtrain_environ_tensor = torch.tensor(xtrain_environ, dtype=torch.float32)
    ytrain_tensor = torch.tensor(ytrain.values, dtype=torch.float32)
    envs_train_tensor = torch.tensor(xtrain_envs, dtype=torch.float32)
    locs_train_tensor = torch.tensor(xtrain_locs, dtype=torch.float32)
    xval_dna_tensor = torch.tensor(xval_dna, dtype=torch.float32)
    # xval_environ_tensor = torch.tensor(xval_environ, dtype=torch.float32)
    yval_tensor = torch.tensor(yval.values, dtype=torch.float32)
    envs_val_tensor = torch.tensor(xval_envs, dtype=torch.float32)
    locs_val_tensor = torch.tensor(xval_locs, dtype=torch.float32)

    # define loss
    mse = nn.MSELoss()
    # cos = nn.CosineSimilarity()

    if MODEL == "G":
        train_dna_dataset = TensorDataset(
            xtrain_dna_tensor, ytrain_tensor, envs_train_tensor
        )
        val_dna_dataset = TensorDataset(xval_dna_tensor, yval_tensor, envs_val_tensor)
        train_dna_loader = DataLoader(
            train_dna_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_dna_loader = DataLoader(
            val_dna_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        input_dim = xtrain_dna.shape[1]
        model = DNA_Net(input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = set_scheduler(optimizer, GAMMA)
        train(
            model,
            optimizer,
            scheduler,
            train_dna_loader,
            val_dna_loader,
            EPOCHS,
            PATIENCE,
        )
        pred_tensor = xval_dna_tensor

    else:
        raise Exception(f"Model {MODEL} not implemented.")

    # predict on validation set and evaluate
    model.load_state_dict(
        torch.load(
            f"output/best_model_{MODEL}_{LOSS}_{BATCH_SIZE}_{GAMMA}_fl_{FILTER_LOC}_device_{DEVICE}.pt",
            weights_only=True,
            map_location=device
        )
    )
    pred_val = xval.index.to_frame(index=False)
    pred_val["ytrue"] = yval.values
    pred_val["ypred"] = predict(model, pred_tensor)
    corr = (
        pred_val.groupby("Env")[["ytrue", "ypred"]]
        .corr()
        .iloc[::2, 1]
        .droplevel(1)
        .sort_values(ascending=True)
    )
    print("corr:\n", corr)
    value = corr.mean()
    print("mean_r:", value)

    # save predictions
    pred_val.to_csv(
        f"output/pred_nn_{MODEL}_{LOSS}_{BATCH_SIZE}_{GAMMA}_fl_{FILTER_LOC}_device_{DEVICE}.csv",
        index=False,
    )

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import datetime
import time
import yaml
import json
import sys
import copy

sys.path.append("..")
from lib.utils import print_log, seed_everything, set_cpu_num, CustomJSONEncoder
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_train_dataloader
from model.MPTSAformer import MPTSAformer

from torch.nn.parallel import DataParallel

@torch.no_grad()
def eval_model(model, device, valset_loader, scaler, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch, long_history_batch in valset_loader:
        x_batch[..., 0] = scaler.transform(x_batch[..., 0])
        long_history_batch[..., 0] = scaler.transform(long_history_batch[..., 0])
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        long_history_batch = long_history_batch.to(device)

        out_batch = model(x_batch, long_history_batch)
        out_batch = scaler.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, device, testset_loader, scaler):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch, long_history_batch in testset_loader:
        x_batch[..., 0] = scaler.transform(x_batch[..., 0])
        long_history_batch[..., 0] = scaler.transform(long_history_batch[..., 0])
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        long_history_batch = long_history_batch.to(device)

        out_batch = model(x_batch, long_history_batch)
        out_batch = scaler.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(model, device, trainset_loader, scaler, optimizer, scheduler, criterion):
    model.train()
    batch_loss_list = []

    for x_batch, y_batch, long_history_batch in trainset_loader:
        x_batch[..., 0] = scaler.transform(x_batch[..., 0])
        long_history_batch[..., 0] = scaler.transform(long_history_batch[..., 0])
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        long_history_batch = long_history_batch.to(device)

        out_batch = model(x_batch, long_history_batch)
        out_batch = scaler.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss


def train(
    model,
    device,
    trainset_loader,
    valset_loader,
    scaler,
    optimizer,
    scheduler,
    criterion,
    max_epochs,
    early_stop,
    verbose,
    multi_gpu,
    save=None,
    log=None,
):
    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, device, trainset_loader, scaler, optimizer, scheduler, criterion)
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, device, valset_loader, scaler, criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                f" \tTrain Loss = {train_loss:.5f}",
                f"Val Loss = {val_loss:.5f}",
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch

            if multi_gpu:
                best_state_dict = copy.deepcopy(model.module.state_dict())
            else:
                best_state_dict = copy.deepcopy(model.state_dict())

        else:
            wait += 1
            if wait >= early_stop:
                break

    if save:
        torch.save(best_state_dict, save)

    if multi_gpu:
        model.module.load_state_dict(best_state_dict)
    else:
        model.load_state_dict(best_state_dict)
    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, device, trainset_loader, scaler))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, device, valset_loader, scaler))

    out_str = f"Early stopping at epoch: {epoch + 1}\n"
    out_str += f"Best at epoch {best_epoch + 1}:\n"
    out_str += f"Train Loss = {train_loss_list[best_epoch]:.5f}\n"
    out_str += f"Train RMSE = {train_rmse:.5f}, MAE = {train_mae:.5f}, MAPE = {train_mape:5f}\n"
    out_str += f"Val Loss = {val_loss_list[best_epoch]:.5f}\n"
    out_str += f"Val RMSE = {val_rmse:.5f}, MAE = {val_mae:.5f}, MAPE = {val_mape:.5f}"
    print_log(out_str, log=log)

    return model


@torch.no_grad()
def test_model(model, device, testset_loader, scaler, log):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, device, testset_loader, scaler)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = f"All Steps RMSE = {rmse_all:.5f}, MAE = {mae_all:.5f}, MAPE = {mape_all:.5f}\n"
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += f"Step {i + 1} RMSE = {rmse:.5f}, MAE = {mae:.5f}, MAPE = {mape:.5f}\n"

    print_log(out_str, log=log, end="")
    print_log(f"Inference time: {(end - start):.2f} s", log=log)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PEMS08")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--muli_gpu", type=int, default=0)
    parser.add_argument("--gpu_list", type=list, default=[0, 1, 2])
    parser.add_argument("--gpu_num", type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    seed_everything(seed)
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    data_path = f"../data/{args.dataset}"

    with open("MPTSAformer.yaml", "r") as f:
        MPTSAformer_cfg = yaml.safe_load(f)
    MPTSAformer_cfg = MPTSAformer_cfg[args.dataset]

    # -------------------------------- load model -------------------------------- #

    model = MPTSAformer(args.dataset, device, MPTSAformer_cfg["TSAformer_args"], MPTSAformer_cfg['pretrain_length'], MPTSAformer_cfg['mask_ratio'])
    model = model.to(device)
    if args.muli_gpu:
        model = DataParallel(model, device_ids=args.gpu_list)

    # ------------------------------- make log file ------------------------------ #

    model_name = "MPTSAformer"
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = "../logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{args.dataset}-{MPTSAformer_cfg['pretrain_length']}-{MPTSAformer_cfg['mask_ratio']}-{now}.log")
    log = open(log, "a", encoding="utf-8")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log(args.dataset, log=log)
    trainset_loader, valset_loader, testset_loader, scaler = get_train_dataloader(
        data_path,
        tod=MPTSAformer_cfg["time_of_day"],
        dow=MPTSAformer_cfg["day_of_week"],
        batch_size=MPTSAformer_cfg["batch_size"],
        in_steps=MPTSAformer_cfg["in_steps"],
        out_steps=MPTSAformer_cfg["out_steps"],
        pretrain_length=MPTSAformer_cfg["pretrain_length"],
    )

    # --------------------------- set model saving path -------------------------- #

    save_path = "../saved_model/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{args.dataset}-{MPTSAformer_cfg['pretrain_length']}-{MPTSAformer_cfg['mask_ratio']}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MPTSAformer_cfg["lr"], weight_decay=MPTSAformer_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=MPTSAformer_cfg["milestones"], gamma=MPTSAformer_cfg["lr_decay_rate"], verbose=False
    )

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(json.dumps(MPTSAformer_cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log)
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model = train(
        model,
        device,
        trainset_loader,
        valset_loader,
        scaler,
        optimizer,
        scheduler,
        criterion,
        max_epochs=MPTSAformer_cfg["max_epochs"],
        early_stop=MPTSAformer_cfg["early_stop"],
        verbose=1,
        multi_gpu=args.muli_gpu,
        save=save,
        log=log,
    )

    print_log(f"Saved Model: {save}", log=log)

    test_model(model, device, testset_loader, scaler, log=log)

    log.close()

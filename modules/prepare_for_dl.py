import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary

def choose_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

DEVICE = choose_device()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_window_index(dates, L, H, val_start, test_start):
    n = len(dates)
    idx_train, idx_val = [], []
    for t in range(L - 1, n - H):
        horizon_end = pd.Timestamp(dates[t + H])
        if horizon_end < val_start:
            idx_train.append(t)
        elif horizon_end < test_start:
            idx_val.append(t)
    t_test = n - H - 1
    if t_test >= L - 1:
        return idx_train, idx_val, [t_test]
    return idx_train, idx_val, []


class WindowDataset(Dataset):
    def __init__(self, series_data, keys, t_indices_by_key, L, H):
        self.items = []
        self.series_data = series_data
        self.L = L
        self.H = H
        for k in keys:
            for t in t_indices_by_key[k]:
                self.items.append((k, t))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        (k, t) = self.items[i]
        d = self.series_data[k]
        y = d["y"]
        known = d["known"]
        past_cov = d["past_cov"]

        x_y = y[t - self.L + 1 : t + 1]  # (L, )
        x_past_cov = past_cov[t - self.L + 1 : t + 1, :]  # (L, Cp)
        x_known_past = known[t - self.L + 1 : t + 1, :]  # (L, Cf)
        x_known_fut = known[t + 1 : t + self.H + 1, :]  # (H, Cf)

        y_fut = y[t + 1 : t + self.H + 1]  # (H, )

        return {
            "x_y": torch.tensor(x_y).float(),  # (L, )
            "x_past_cov": torch.tensor(x_past_cov).float(),  # (L, Cp)
            "x_known_past": torch.tensor(x_known_past).float(),  # (L, Cf)
            "x_known_fut": torch.tensor(x_known_fut).float(),  # (H, Cf)
            "y": torch.tensor(y_fut).float(),  # (H, )
            "store_id": torch.tensor(d["store_id"]).long(),
        }


def collate(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def rmse_torch(y, yhat, eps=1e-6):
    return torch.sqrt(((yhat - y) ** 2).mean() + eps)


def rmse_numpy(y, yhat, eps=1e-6):
    return np.sqrt(np.mean((yhat - y) ** 2) + eps)


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def validation_loop(model, dl, return_preds=False, quantile_idx=1):
    model.eval()

    losses, rmses = [], []
    if return_preds:
        all_y_true = []
        all_y_pred = []
        all_store = []
        all_ctx = []

    for batch in dl:
        batch = to_device(batch, DEVICE)

        y = batch["y"]  # (B, H)
        out = model(batch)

        if out.ndim == 3 and out.shape[-1] == 3:
            yhat = out[:, :, quantile_idx]
        else:
            yhat = out  # (B, H)

        loss = rmse_torch(yhat, y)
        sm = rmse_torch(y, yhat)

        losses.append(loss.item())
        rmses.append(sm.item())

        if return_preds:
            all_y_true.append(y.detach().cpu())
            all_y_pred.append(yhat.detach().cpu())
            all_store.append(batch["store_id"].detach().cpu())
            all_ctx.append(batch["x_y"].detach().cpu())

    mean_loss = float(np.mean(losses))
    mean_sm = float(np.mean(rmses))

    if not return_preds:
        return mean_loss, mean_sm

    pack = {
        "y_true": torch.cat(all_y_true, dim=0).numpy(),  # (N, H)
        "y_pred": torch.cat(all_y_pred, dim=0).numpy(),  # (N, H)
        "store_id": torch.cat(all_store, dim=0).numpy(),  # (N, )
        "x_y": torch.cat(all_ctx, dim=0).numpy(),  # (N, L)
    }
    return mean_loss, mean_sm, pack


def train_loop(model, train_dl, val_dl, epochs=5, lr=1e-3, wd=1e-4, max_grad=1.0):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_dl:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)

            y = batch["y"]
            yhat = model(batch)
            loss = rmse_torch(yhat, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            opt.step()

        tr_loss, tr_sm = validation_loop(model, train_dl)
        va_loss, va_sm = validation_loop(model, val_dl)

        print(
            f"Epoch {ep:02d} | train loss {tr_loss:.4f} rmse {tr_sm:.4f} | val loss {va_loss:.4f} rmse {va_sm:.4f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def build_stats_from_series_data(series_data):
    stats = {}
    for _, d in series_data.items():
        stats[(int(d["store_id"]))] = (float(d["y_mu"]), float(d["y_sd"]))
    return stats


def denorm_to_sales(y_norm, mu, sd):
    return np.expm1(y_norm * sd + mu)


def _select_point_pred(y_pred, mode="median", quantile_index=1):
    if y_pred.ndim == 2:
        return y_pred
    if y_pred.ndim == 3:
        if mode == "median":
            return y_pred[:, :, quantile_index]
        if mode == "mean":
            return y_pred.mean(axis=-1)


def denorm_forecasts(preds, series_data, point_mode="median", quantile_index=1):
    stats = build_stats_from_series_data(series_data)

    y_true = preds["y_true"]
    y_pred = _select_point_pred(
        preds["y_pred"], mode=point_mode, quantile_index=quantile_index
    )

    store_ids = preds["store_id"].astype(int)

    N, H = y_true.shape
    y_true_sales = np.empty((N, H), dtype=np.float32)
    y_pred_sales = np.empty((N, H), dtype=np.float32)

    for i in range(N):
        mu, sd = stats[(store_ids[i])]
        y_true_sales[i] = denorm_to_sales(y_true[i], mu, sd)
        y_pred_sales[i] = denorm_to_sales(y_pred[i], mu, sd)

    return y_true_sales, y_pred_sales

def quantile_loss(yhat, y, qs=(0.1, 0.5, 0.9)):
    y = y.unsqueeze(-1)
    q = torch.tensor(qs, device=y.device).view(1, 1, -1)
    e = y - yhat
    return torch.maximum(q * e, (q - 1) * e).mean()

def train_loop_tft(model, train_dl, val_dl, epochs=10, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
    
    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_dl:
            batch = to_device(batch, DEVICE)
            opt.zero_grad()
            loss = quantile_loss(model(batch), batch["y"])
            loss.backward()
            opt.step()
        
        v_loss, v_rmse = validation_loop(model, val_dl, quantile_idx=1)
        print(f"Epoch {ep:02d} | Val RMSE: {v_rmse:.4f}")
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_state)
    return model

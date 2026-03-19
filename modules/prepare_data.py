import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoETS, AutoTheta
import random

import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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


def transform_data(train, oil, items):
    ds = train[(train["item_nbr"] == 502331)]

    start_date = "2017-01-01"
    end_date = "2017-08-15"

    dates = pd.date_range(start=start_date, end=end_date)

    tiled_dates = np.tile(dates, 54)
    repeated_dates = pd.DataFrame({"date_r": tiled_dates})

    result = pd.DataFrame(np.repeat([i for i in range(1, 55)], 227))
    repeated_dates["store"] = result

    df_502331 = repeated_dates.merge(
        ds, left_on=["date_r", "store"], right_on=["date", "store_nbr"], how="left"
    )
    df_502331.drop(["date", "store_nbr"], axis=1, inplace=True)

    for i in range(len(df_502331["date_r"])):
        if np.isnan(df_502331["unit_sales"][i]):
            df_502331["unit_sales"][i] = 0

    df_502331["onpromotion"][0] = False

    for i in range(1, len(df_502331["date_r"]) - 1):
        if np.isnan(df_502331["onpromotion"][i]):
            if (np.isnan(df_502331["onpromotion"][i - 1]) == False) & (
                df_502331["store"][i - 1] == df_502331["store"][i]
            ):
                df_502331["onpromotion"][i] = df_502331["onpromotion"][i - 1]
            elif (np.isnan(df_502331["onpromotion"][i + 1]) == False) & (
                df_502331["store"][i + 1] == df_502331["store"][i]
            ):
                df_502331["onpromotion"][i] = df_502331["onpromotion"][i + 1]
            else:
                df_502331["onpromotion"][i] = True

    df = df_502331[["date_r", "store", "item_nbr", "unit_sales", "onpromotion"]].copy()

    data = df[["date_r", "store", "item_nbr", "unit_sales", "onpromotion"]].copy()
    data = data.rename(columns={"date_r": "ds", "unit_sales": "y"})

    data = data.sort_values(["store", "ds"])

    data = data.merge(items, left_on="item_nbr", right_on="item_nbr", how="left")

    data["ds"] = pd.to_datetime(data["ds"])
    oil["date"] = pd.to_datetime(oil["date"])

    oil = oil[oil["date"] >= "2017-01-01"]

    oil_transf = pd.DataFrame(dates, columns=["d"])

    oil_transf = oil_transf.merge(oil, left_on="d", right_on="date", how="left")
    oil_transf.drop("date", axis=1, inplace=True)

    oil_transf["dcoilwtico"][0] = 52.36
    oil_transf["dcoilwtico"][1] = 52.36

    for i in range(len(oil_transf["d"])):
        if np.isnan(oil_transf["dcoilwtico"][i]):
            oil_transf["dcoilwtico"][i] = oil_transf["dcoilwtico"][i - 1]

    data = data.merge(oil_transf, left_on="ds", right_on="d", how="left")
    data = data.drop(columns=["d"])

    data = data.sort_values(["store", "ds"]).reset_index(drop=True)

    data.drop(["item_nbr", "family", "class", "perishable"], axis=1, inplace=True)

    data["onpromotion"] = data["onpromotion"].astype(str) == "True"

    return data
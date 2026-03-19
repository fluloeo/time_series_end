import pandas as pd
import numpy as np


def calculate_rmsle(
    group: pd.DataFrame,
    pred_column: str = "predicted_value",
    true_column: str = "true_value",
) -> pd.Series:

    y_true = group[true_column].values
    y_pred = group[pred_column].values

    y_pred = np.maximum(0, y_pred)

    log_diff = np.log1p(y_pred) - np.log1p(y_true)

    rmsle_value = np.sqrt(np.mean(np.square(log_diff)))

    return pd.Series({"RMSLE": rmsle_value})

import numpy as np

def nwrmsle(y_true, y_pred, weights=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred = np.clip(y_pred, 0, None)
    log_diff = np.log1p(y_pred) - np.log1p(y_true)

    if weights is None:
        weights = np.ones_like(y_true)

    return np.sqrt(np.sum(weights * log_diff**2) / np.sum(weights))

def rmsle(y_true, y_pred):
     y_true = np.clip(y_true, 0, None)
     y_pred = np.clip(y_pred, 0, None)
     valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
     if not valid.any():
         return np.nan
     return np.sqrt(np.mean((np.log1p(y_true[valid]) - np.log1p(y_pred[valid])) ** 2))
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    abs_err = np.abs(y_true - y_pred)
    rel = abs_err / np.maximum(np.abs(y_true), 1e-8)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "mre": float(np.mean(rel)),
        "medre": float(np.median(rel)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

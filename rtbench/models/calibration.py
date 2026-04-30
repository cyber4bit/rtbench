from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..metrics import compute_metrics
from .trees import CandidateOutput


def optimize_weights(
    y_val: np.ndarray,
    pred_mat: np.ndarray,
    objective: str = "mae",
    r2_weight: float = 0.25,
    l2_reg: float = 0.0,
    weight_floor: float = 0.0,
) -> np.ndarray:
    k = pred_mat.shape[1]
    x0 = np.full(k, 1.0 / k, dtype=np.float64)
    y = np.asarray(y_val, dtype=np.float64)
    var_y = float(np.var(y)) + 1e-12
    std_y = float(np.std(y)) + 1e-12
    obj_mode = str(objective or "mae").strip().lower()
    r2_w = float(np.clip(r2_weight, 0.0, 1.0))
    reg = float(max(l2_reg, 0.0))
    floor = float(max(weight_floor, 0.0))
    if k > 0 and floor * k >= 1.0:
        floor = max(0.0, (1.0 / k) - 1e-6)

    def obj(w: np.ndarray) -> float:
        pred = pred_mat @ w
        err = y - pred
        mae = float(np.mean(np.abs(err)))
        if obj_mode in ("mae_r2", "mae+mse", "balanced"):
            mae_norm = mae / std_y
            mse_norm = float(np.mean(err * err) / var_y)
            base = float((1.0 - r2_w) * mae_norm + r2_w * mse_norm)
        else:
            base = mae
        if reg > 0.0:
            base += reg * float(np.mean((w - (1.0 / k)) ** 2))
        return float(base)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(floor, 1.0)] * k
    res = minimize(obj, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
    if not res.success:
        return x0
    w = np.clip(res.x, 0.0, 1.0)
    s = float(np.sum(w))
    return w / s if s > 0 else x0


def calibrate_linear(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(y) < 2 or np.allclose(np.var(pred), 0.0):
        return 1.0, 0.0
    a, b = np.polyfit(pred, y, deg=1)
    if not np.isfinite(a) or not np.isfinite(b):
        return 1.0, 0.0
    a = float(np.clip(a, 0.1, 10.0))
    b = float(np.clip(b, -10.0 * np.std(y), 10.0 * np.std(y)))
    return a, b


def apply_calibration(a: float, b: float, pred: np.ndarray) -> np.ndarray:
    return (a * np.asarray(pred, dtype=np.float64) + b).astype(np.float32)


def calibrate_candidates(candidates: list[CandidateOutput], y_val_sec: np.ndarray) -> None:
    for candidate in candidates:
        a, b = calibrate_linear(y_val_sec, candidate.val_pred)
        candidate.val_pred = apply_calibration(a, b, candidate.val_pred)
        candidate.test_pred = apply_calibration(a, b, candidate.test_pred)
        candidate.val_metrics = compute_metrics(y_val_sec, candidate.val_pred)

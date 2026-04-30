from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..metrics import compute_metrics
from .trees import CandidateOutput, _inverse_target


def _fit_ridge_models(
    model_cfg: dict[str, dict[str, Any]],
    X_src: np.ndarray,
    y_src: np.ndarray,
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    X_val: np.ndarray,
    *,
    y_val_used: np.ndarray | None = None,
    y_val_sec: np.ndarray | None = None,
    X_test: np.ndarray,
    seed: int,
    source_weight: float,
    target_weight: float,
    source_sample_weights: np.ndarray | None = None,
    name_prefix: str = "",
    target_transform: str = "none",
    target_inv_scale: float = 1.0,
    target_t0_sec: float = 1.0,
) -> list[CandidateOutput]:
    if y_val_sec is None:
        if y_val_used is None:
            raise ValueError("y_val_sec or y_val_used is required in _fit_ridge_models")
        y_val_sec = np.asarray(y_val_used, dtype=np.float32)
    alpha_tl = float(model_cfg.get("RIDGE_TL_ALPHA", 10.0))
    alpha_local = float(model_cfg.get("RIDGE_LOCAL_ALPHA", 10.0))

    outputs: list[CandidateOutput] = []

    # Transfer ridge (source + target train with weights).
    X_train = np.concatenate([X_src, X_t_train], axis=0)
    y_train = np.concatenate([y_src, y_t_train], axis=0)
    if source_sample_weights is None:
        src_w = np.full(len(X_src), source_weight, dtype=np.float32)
    else:
        src_w = np.asarray(source_sample_weights, dtype=np.float32)
        if len(src_w) != len(X_src):
            raise ValueError("source_sample_weights length mismatch in _fit_ridge_models")
    w = np.concatenate([src_w, np.full(len(X_t_train), target_weight, dtype=np.float32)])

    xs = StandardScaler().fit(X_train)
    X_train_s = xs.transform(X_train).astype(np.float32)
    X_val_s = xs.transform(X_val).astype(np.float32)
    X_test_s = xs.transform(X_test).astype(np.float32)

    rt = Ridge(alpha=alpha_tl, random_state=seed)
    rt.fit(X_train_s, y_train, sample_weight=w)
    val_pred_used = rt.predict(X_val_s)
    y_val_refit = np.asarray(y_val_used if y_val_used is not None else y_val_sec, dtype=np.float32)
    X_refit = np.concatenate([X_train, X_val], axis=0)
    y_refit = np.concatenate([y_train, y_val_refit], axis=0)
    w_refit = np.concatenate([w, np.full(len(X_val), target_weight, dtype=np.float32)])
    xs_refit = StandardScaler().fit(X_refit)
    rt_final = Ridge(alpha=alpha_tl, random_state=seed)
    rt_final.fit(xs_refit.transform(X_refit).astype(np.float32), y_refit, sample_weight=w_refit)
    test_pred_used = rt_final.predict(xs_refit.transform(X_test).astype(np.float32))
    val_pred = _inverse_target(val_pred_used, target_transform, target_inv_scale, target_t0_sec)
    test_pred = _inverse_target(test_pred_used, target_transform, target_inv_scale, target_t0_sec)
    outputs.append(
        CandidateOutput(
            name="RIDGE_TL",
            val_pred=val_pred,
            test_pred=test_pred,
            val_metrics=compute_metrics(y_val_sec, val_pred),
            model=rt_final,
        )
    )

    # Local ridge (target train only).
    xs2 = StandardScaler().fit(X_t_train)
    X_t_s = xs2.transform(X_t_train).astype(np.float32)
    X_val_s2 = xs2.transform(X_val).astype(np.float32)
    X_test_s2 = xs2.transform(X_test).astype(np.float32)
    rl = Ridge(alpha=alpha_local, random_state=seed + 101)
    rl.fit(X_t_s, y_t_train, sample_weight=np.full(len(X_t_train), target_weight, dtype=np.float32))
    val_pred_used = rl.predict(X_val_s2)
    X_t_refit = np.concatenate([X_t_train, X_val], axis=0)
    y_t_refit = np.concatenate([y_t_train, y_val_refit], axis=0)
    w_t_refit = np.full(len(X_t_refit), target_weight, dtype=np.float32)
    xs2_refit = StandardScaler().fit(X_t_refit)
    rl_final = Ridge(alpha=alpha_local, random_state=seed + 101)
    rl_final.fit(xs2_refit.transform(X_t_refit).astype(np.float32), y_t_refit, sample_weight=w_t_refit)
    test_pred_used = rl_final.predict(xs2_refit.transform(X_test).astype(np.float32))
    val_pred = _inverse_target(val_pred_used, target_transform, target_inv_scale, target_t0_sec)
    test_pred = _inverse_target(test_pred_used, target_transform, target_inv_scale, target_t0_sec)
    outputs.append(
        CandidateOutput(
            name="RIDGE_LOCAL",
            val_pred=val_pred,
            test_pred=test_pred,
            val_metrics=compute_metrics(y_val_sec, val_pred),
            model=rl_final,
        )
    )

    if name_prefix:
        for out in outputs:
            out.name = f"{name_prefix}{out.name}"
    return outputs

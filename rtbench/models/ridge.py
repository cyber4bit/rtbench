from __future__ import annotations

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
    y_val_sec: np.ndarray,
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
    test_pred_used = rt.predict(X_test_s)
    val_pred = _inverse_target(val_pred_used, target_transform, target_inv_scale, target_t0_sec)
    test_pred = _inverse_target(test_pred_used, target_transform, target_inv_scale, target_t0_sec)
    outputs.append(
        CandidateOutput(
            name="RIDGE_TL",
            val_pred=val_pred,
            test_pred=test_pred,
            val_metrics=compute_metrics(y_val_sec, val_pred),
            model=rt,
        )
    )

    # Local ridge (target train only).
    xs2 = StandardScaler().fit(X_t_train)
    X_t_s = xs2.transform(X_t_train).astype(np.float32)
    X_val_s2 = xs2.transform(X_val).astype(np.float32)
    X_test_s2 = xs2.transform(X_test).astype(np.float32)
    rl = Ridge(alpha=alpha_local, random_state=seed + 101)
    rl.fit(X_t_s, y_t_train)
    val_pred_used = rl.predict(X_val_s2)
    test_pred_used = rl.predict(X_test_s2)
    val_pred = _inverse_target(val_pred_used, target_transform, target_inv_scale, target_t0_sec)
    test_pred = _inverse_target(test_pred_used, target_transform, target_inv_scale, target_t0_sec)
    outputs.append(
        CandidateOutput(
            name="RIDGE_LOCAL",
            val_pred=val_pred,
            test_pred=test_pred,
            val_metrics=compute_metrics(y_val_sec, val_pred),
            model=rl,
        )
    )

    if name_prefix:
        for out in outputs:
            out.name = f"{name_prefix}{out.name}"
    return outputs

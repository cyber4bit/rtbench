from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from ..data import FINGERPRINT_SIZES
from ..metrics import compute_metrics


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class CandidateOutput:
    name: str
    val_pred: np.ndarray
    test_pred: np.ndarray
    val_metrics: dict[str, float]
    model: Any


def stratified_split(y: np.ndarray, seed: int, train: float, val: float, test: float) -> SplitData:
    if abs((train + val + test) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to 1.0")
    y = np.asarray(y)
    if len(y) < 2:
        raise ValueError("stratified_split requires at least 2 samples")
    n_bins = min(10, max(2, len(y) // 8))
    bins = pd.qcut(y, q=n_bins, duplicates="drop")
    idx = np.arange(len(y))
    try:
        train_idx, temp_idx = train_test_split(
            idx,
            test_size=(1.0 - train),
            random_state=seed,
            stratify=bins,
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(
            idx,
            test_size=(1.0 - train),
            random_state=seed,
            stratify=None,
        )
    temp_y = y[temp_idx]
    temp_bins = pd.qcut(temp_y, q=min(5, max(2, len(temp_y) // 5)), duplicates="drop")
    test_frac_within_temp = test / (val + test)
    try:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_frac_within_temp,
            random_state=seed + 1,
            stratify=temp_bins,
        )
    except ValueError:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_frac_within_temp,
            random_state=seed + 1,
            stratify=None,
        )
    return SplitData(train_idx=np.array(train_idx), val_idx=np.array(val_idx), test_idx=np.array(test_idx))


def random_split(y: np.ndarray, seed: int, train: float, val: float, test: float) -> SplitData:
    if abs((train + val + test) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to 1.0")
    n = int(len(y))
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    tv = int(n * (train + val))
    tr = int(tv * (train / max(train + val, 1e-8)))
    train_idx = idx[:tr]
    val_idx = idx[tr:tv]
    test_idx = idx[tv:]
    return SplitData(train_idx=np.array(train_idx), val_idx=np.array(val_idx), test_idx=np.array(test_idx))


def _normalize_target_transform(name: str) -> str:
    n = str(name or "none").strip().lower()
    if n in ("", "none", "identity"):
        return "none"
    if n in ("gradient_norm", "gradnorm", "gradient_normalize", "target_normalize", "normalize"):
        return "gradient_norm"
    if n in ("logk", "log_k", "logretention", "log_retention"):
        return "logk"
    if n in ("log1p", "log_rt", "logrt"):
        return "log1p"
    return "none"


def _inverse_target(pred: np.ndarray, target_transform: str, inv_scale: float, t0_sec: float) -> np.ndarray:
    """Inverse-transform model outputs back to seconds for metric computation/output."""
    tfm = _normalize_target_transform(target_transform)
    pred64 = np.asarray(pred, dtype=np.float64)
    if tfm == "gradient_norm":
        out = pred64 * float(inv_scale)
    elif tfm == "logk":
        # Avoid exp overflow producing inf; extreme values will be clipped again downstream.
        pred64 = np.clip(pred64, -50.0, 50.0)
        out = float(t0_sec) * (np.exp(pred64) + 1.0)
    elif tfm == "log1p":
        pred64 = np.clip(pred64, -50.0, 50.0)
        out = np.expm1(pred64)
    else:
        out = pred64
    return out.astype(np.float32)


def _forward_target(y_sec: np.ndarray, target_transform: str, inv_scale: float, t0_sec: float) -> np.ndarray:
    """Forward-transform seconds into model target space."""
    tfm = _normalize_target_transform(target_transform)
    y64 = np.asarray(y_sec, dtype=np.float64)
    if tfm == "gradient_norm":
        out = y64 / max(float(inv_scale), 1e-6)
    elif tfm == "logk":
        t0 = max(float(t0_sec), 1e-6)
        out = np.log(np.clip((y64 - t0) / t0, 1e-6, None))
    elif tfm == "log1p":
        out = np.log1p(np.clip(y64, 0.0, None))
    else:
        out = y64
    return out.astype(np.float32)


def _fit_tree_models(
    model_cfg: dict[str, dict[str, Any]],
    X_src: np.ndarray,
    y_src: np.ndarray,
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    X_val: np.ndarray,
    y_val_used: np.ndarray,
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
    X_train = np.concatenate([X_src, X_t_train], axis=0)
    y_train = np.concatenate([y_src, y_t_train], axis=0)
    if source_sample_weights is None:
        src_w = np.full(len(X_src), source_weight, dtype=np.float32)
    else:
        src_w = np.asarray(source_sample_weights, dtype=np.float32)
        if len(src_w) != len(X_src):
            raise ValueError("source_sample_weights length mismatch in _fit_tree_models")
    w = np.concatenate([src_w, np.full(len(X_t_train), target_weight, dtype=np.float32)])

    models: list[tuple[str, Any]] = []
    xgb_a = XGBRegressor(random_state=seed, n_jobs=8, verbosity=0, **model_cfg["XGB_A"])
    xgb_b = XGBRegressor(random_state=seed + 7, n_jobs=8, verbosity=0, **model_cfg["XGB_B"])
    lgbm_a = LGBMRegressor(random_state=seed + 13, n_jobs=8, verbose=-1, **model_cfg["LGBM_A"])
    lgbm_b = LGBMRegressor(random_state=seed + 19, n_jobs=8, verbose=-1, **model_cfg["LGBM_B"])
    models.extend([("XGB_A", xgb_a), ("XGB_B", xgb_b), ("LGBM_A", lgbm_a), ("LGBM_B", lgbm_b)])

    outputs: list[CandidateOutput] = []
    lgbm_cols = [f"f{i}" for i in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=lgbm_cols)
    X_val_df = pd.DataFrame(X_val, columns=lgbm_cols)
    X_test_df = pd.DataFrame(X_test, columns=lgbm_cols)
    stop_rounds = int(model_cfg.get("EARLY_STOPPING_ROUNDS", 0))
    for name, model in models:
        if name.startswith("LGBM"):
            callbacks = [early_stopping(stopping_rounds=stop_rounds, verbose=False)] if stop_rounds > 0 else None
            if callbacks:
                model.fit(
                    X_train_df,
                    y_train,
                    sample_weight=w,
                    eval_set=[(X_val_df, y_val_used)],
                    eval_metric="l1",
                    callbacks=callbacks,
                )
                num_it = int(getattr(model, "best_iteration_", 0)) or None
                val_pred_used = model.predict(X_val_df, num_iteration=num_it)
                test_pred_used = model.predict(X_test_df, num_iteration=num_it)
            else:
                model.fit(X_train_df, y_train, sample_weight=w)
                val_pred_used = model.predict(X_val_df)
                test_pred_used = model.predict(X_test_df)
        else:
            if stop_rounds > 0:
                # XGBoost 3.x: early stopping / eval metric are parameters, not fit() kwargs.
                try:
                    model.set_params(early_stopping_rounds=stop_rounds, eval_metric="mae")
                except Exception:
                    pass
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=w,
                    eval_set=[(X_val, y_val_used)],
                    verbose=False,
                )
                best_it = getattr(model, "best_iteration", None)
                if best_it is not None:
                    val_pred_used = model.predict(X_val, iteration_range=(0, int(best_it) + 1))
                    test_pred_used = model.predict(X_test, iteration_range=(0, int(best_it) + 1))
                else:
                    val_pred_used = model.predict(X_val)
                    test_pred_used = model.predict(X_test)
            else:
                model.fit(X_train, y_train, sample_weight=w)
                val_pred_used = model.predict(X_val)
                test_pred_used = model.predict(X_test)
        val_pred = _inverse_target(val_pred_used, target_transform, target_inv_scale, target_t0_sec)
        test_pred = _inverse_target(test_pred_used, target_transform, target_inv_scale, target_t0_sec)
        outputs.append(
            CandidateOutput(
                name=name,
                val_pred=val_pred,
                test_pred=test_pred,
                val_metrics=compute_metrics(y_val_sec, val_pred),
                model=model,
            )
        )
    if name_prefix:
        for out in outputs:
            out.name = f"{name_prefix}{out.name}"
    return outputs


def _fit_branch_tree_models(
    branch_name: str,
    specs: list[tuple[str, str, dict[str, Any]]],
    X_src: np.ndarray,
    y_src: np.ndarray,
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    X_val: np.ndarray,
    y_val_used: np.ndarray,
    y_val_sec: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    source_weight: float,
    target_weight: float,
    early_stopping_rounds: int = 0,
    source_sample_weights: np.ndarray | None = None,
    target_transform: str = "none",
    target_inv_scale: float = 1.0,
    target_t0_sec: float = 1.0,
) -> list[CandidateOutput]:
    X_train = np.concatenate([X_src, X_t_train], axis=0)
    y_train = np.concatenate([y_src, y_t_train], axis=0)
    if source_sample_weights is None:
        src_w = np.full(len(X_src), source_weight, dtype=np.float32)
    else:
        src_w = np.asarray(source_sample_weights, dtype=np.float32)
        if len(src_w) != len(X_src):
            raise ValueError("source_sample_weights length mismatch in _fit_branch_tree_models")
    w = np.concatenate([src_w, np.full(len(X_t_train), target_weight, dtype=np.float32)])

    outputs: list[CandidateOutput] = []
    lgbm_cols = [f"f{i}" for i in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=lgbm_cols)
    X_val_df = pd.DataFrame(X_val, columns=lgbm_cols)
    X_test_df = pd.DataFrame(X_test, columns=lgbm_cols)
    stop_rounds = int(early_stopping_rounds)

    for idx, (name, algo, params) in enumerate(specs):
        if algo == "xgb":
            model = XGBRegressor(random_state=seed + idx * 17, n_jobs=8, verbosity=0, **params)
            if stop_rounds > 0:
                try:
                    model.set_params(early_stopping_rounds=stop_rounds, eval_metric="mae")
                except Exception:
                    pass
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=w,
                    eval_set=[(X_val, y_val_used)],
                    verbose=False,
                )
                best_it = getattr(model, "best_iteration", None)
                if best_it is not None:
                    val_pred_used = model.predict(X_val, iteration_range=(0, int(best_it) + 1))
                    test_pred_used = model.predict(X_test, iteration_range=(0, int(best_it) + 1))
                else:
                    val_pred_used = model.predict(X_val)
                    test_pred_used = model.predict(X_test)
            else:
                model.fit(X_train, y_train, sample_weight=w)
                val_pred_used = model.predict(X_val)
                test_pred_used = model.predict(X_test)
        elif algo == "lgbm":
            model = LGBMRegressor(random_state=seed + idx * 17, n_jobs=8, verbose=-1, **params)
            callbacks = [early_stopping(stopping_rounds=stop_rounds, verbose=False)] if stop_rounds > 0 else None
            if callbacks:
                model.fit(
                    X_train_df,
                    y_train,
                    sample_weight=w,
                    eval_set=[(X_val_df, y_val_used)],
                    eval_metric="l1",
                    callbacks=callbacks,
                )
                num_it = int(getattr(model, "best_iteration_", 0)) or None
                val_pred_used = model.predict(X_val_df, num_iteration=num_it)
                test_pred_used = model.predict(X_test_df, num_iteration=num_it)
            else:
                model.fit(X_train_df, y_train, sample_weight=w)
                val_pred_used = model.predict(X_val_df)
                test_pred_used = model.predict(X_test_df)
        else:
            raise ValueError(f"Unknown algo '{algo}' in branch '{branch_name}'")
        val_pred = _inverse_target(val_pred_used, target_transform, target_inv_scale, target_t0_sec)
        test_pred = _inverse_target(test_pred_used, target_transform, target_inv_scale, target_t0_sec)
        outputs.append(
            CandidateOutput(
                name=f"{branch_name}_{name}",
                val_pred=val_pred,
                test_pred=test_pred,
                val_metrics=compute_metrics(y_val_sec, val_pred),
                model=model,
            )
        )
    return outputs


def _mdl_feature_subset(X: np.ndarray, group_sizes: dict[str, int]) -> np.ndarray:
    """Paper-style molecular block: descriptors + MACCS, keep all CP features."""
    d = int(group_sizes.get("descriptor", 0))
    f = int(group_sizes.get("fingerprint", 0))
    maccs = int(FINGERPRINT_SIZES.get("maccs", 166))
    if X.ndim != 2 or X.shape[1] <= 0:
        return X
    if d < 0 or f <= 0 or (d + f) > X.shape[1]:
        return X
    maccs = min(max(maccs, 0), f)
    left = X[:, : d + maccs]
    right = X[:, d + f :]
    if right.size == 0:
        return left
    return np.concatenate([left, right], axis=1)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
from lightgbm import early_stopping

from ...hyper import HyperTLBundle
from ..trees import CandidateOutput, _forward_target, _normalize_target_transform

TreeFitFn = Callable[..., list[CandidateOutput]]
RidgeFitFn = Callable[..., list[CandidateOutput]]
MLPTrainFn = Callable[..., CandidateOutput]


@dataclass(frozen=True)
class CandidateBuildContext:
    model_cfg: dict[str, Any]
    X_src: np.ndarray
    X_src_cp: np.ndarray
    y_src: np.ndarray
    y_src_sec: np.ndarray
    X_train: np.ndarray
    X_train_mol: np.ndarray
    X_train_cp: np.ndarray
    y_train: np.ndarray
    y_train_sec: np.ndarray
    X_val: np.ndarray
    X_val_mol: np.ndarray
    y_val_used: np.ndarray
    y_val_sec: np.ndarray
    X_test: np.ndarray
    X_test_mol: np.ndarray
    seed: int
    source_weight: float
    target_weight: float
    group_sizes: dict[str, int]
    fail_tune: bool = False
    source_sample_weights: np.ndarray | None = None
    target_transform: str = "none"
    target_inv_scale: float = 1.0
    target_t0_sec: float = 1.0
    hyper_bundle: HyperTLBundle | list[HyperTLBundle] | None = None
    source_row_dataset_ids: np.ndarray | None = None
    source_mol_keys: np.ndarray | None = None
    target_mol_keys: np.ndarray | None = None
    target_cp_reference: np.ndarray | None = None


def clean_mol_key(value: Any) -> str:
    key = str(value).strip()
    if not key or key in ("nan", "NA", "None"):
        return ""
    return key


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def predict_xgb(
    xgb_regressor_cls: type[Any],
    *,
    params: dict[str, Any],
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val_used: np.ndarray,
    X_test: np.ndarray,
    stop_rounds: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    model = xgb_regressor_cls(random_state=seed, n_jobs=8, verbosity=0, **params)
    if stop_rounds > 0:
        try:
            model.set_params(early_stopping_rounds=stop_rounds, eval_metric="mae")
        except Exception:
            pass
        model.fit(X_train, y_train, eval_set=[(X_val, y_val_used)], verbose=False)
        best_it = getattr(model, "best_iteration", None)
        if best_it is not None:
            val_pred = model.predict(X_val, iteration_range=(0, int(best_it) + 1))
            test_pred = model.predict(X_test, iteration_range=(0, int(best_it) + 1))
        else:
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
    return val_pred, test_pred, model


def predict_lgbm(
    lgbm_regressor_cls: type[Any],
    *,
    params: dict[str, Any],
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val_used: np.ndarray,
    X_test: np.ndarray,
    stop_rounds: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    cols = [f"f{i}" for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train, columns=cols)
    val_df = pd.DataFrame(X_val, columns=cols)
    test_df = pd.DataFrame(X_test, columns=cols)
    model = lgbm_regressor_cls(random_state=seed, n_jobs=8, verbose=-1, **params)
    callbacks = [early_stopping(stopping_rounds=stop_rounds, verbose=False)] if stop_rounds > 0 else None
    if callbacks:
        model.fit(
            train_df,
            y_train,
            eval_set=[(val_df, y_val_used)],
            eval_metric="l1",
            callbacks=callbacks,
        )
        num_it = int(getattr(model, "best_iteration_", 0)) or None
        val_pred = model.predict(val_df, num_iteration=num_it)
        test_pred = model.predict(test_df, num_iteration=num_it)
    else:
        model.fit(train_df, y_train)
        val_pred = model.predict(val_df)
        test_pred = model.predict(test_df)
    return val_pred, test_pred, model


def iter_transfer_target_views(
    ctx: CandidateBuildContext,
) -> Iterator[tuple[int, str, np.ndarray, np.ndarray, np.ndarray]]:
    if not bool(ctx.model_cfg.get("ENABLE_TRANSFER_TRANSFORM_CANDIDATES", False)):
        return
    for index, raw_name in enumerate(ctx.model_cfg.get("TRANSFER_TARGET_TRANSFORMS", ["log1p", "gradient_norm"])):
        tfm_name = _normalize_target_transform(str(raw_name))
        if tfm_name == ctx.target_transform:
            continue
        yield (
            index,
            tfm_name,
            _forward_target(ctx.y_src_sec, tfm_name, ctx.target_inv_scale, ctx.target_t0_sec),
            _forward_target(ctx.y_train_sec, tfm_name, ctx.target_inv_scale, ctx.target_t0_sec),
            _forward_target(ctx.y_val_sec, tfm_name, ctx.target_inv_scale, ctx.target_t0_sec),
        )


def iter_local_target_views(ctx: CandidateBuildContext) -> Iterator[tuple[int, str, np.ndarray, np.ndarray]]:
    if not bool(ctx.model_cfg.get("ENABLE_LOCAL_TRANSFORM_CANDIDATES", False)):
        return
    for index, raw_name in enumerate(ctx.model_cfg.get("LOCAL_TARGET_TRANSFORMS", ["log1p", "gradient_norm"])):
        tfm_name = _normalize_target_transform(str(raw_name))
        if tfm_name == "none":
            continue
        yield (
            index,
            tfm_name,
            _forward_target(ctx.y_train_sec, tfm_name, ctx.target_inv_scale, ctx.target_t0_sec),
            _forward_target(ctx.y_val_sec, tfm_name, ctx.target_inv_scale, ctx.target_t0_sec),
        )

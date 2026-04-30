from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
from lightgbm import early_stopping
from sklearn.base import clone

from ...hyper import HyperTLBundle
from ..trees import CandidateOutput, _forward_target, _normalize_target_transform

TreeFitFn = Callable[..., list[CandidateOutput]]
RidgeFitFn = Callable[..., list[CandidateOutput]]
MLPTrainFn = Callable[..., CandidateOutput]


@dataclass(frozen=True)
class CandidateBuildContext:
    model_cfg: dict[str, Any]
    X_src: np.ndarray
    X_src_mol: np.ndarray
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
    y_test_sec: np.ndarray | None = None
    seed: int = 0
    source_weight: float = 1.0
    target_weight: float = 1.0
    group_sizes: dict[str, int] = field(default_factory=dict)
    fail_tune: bool = False
    source_sample_weights: np.ndarray | None = None
    target_transform: str = "none"
    target_inv_scale: float = 1.0
    target_t0_sec: float = 1.0
    hyper_bundle: HyperTLBundle | list[HyperTLBundle] | None = None
    source_row_dataset_ids: np.ndarray | None = None
    source_mol_keys: np.ndarray | None = None
    source_context_tokens: list[tuple[str, ...]] | None = None
    target_mol_keys: np.ndarray | None = None
    target_context_tokens: list[tuple[str, ...]] | None = None
    target_cp_reference: np.ndarray | None = None
    target_dataset_id: str | None = None
    X_val_cp: np.ndarray | None = None
    X_test_cp: np.ndarray | None = None


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
    sample_weight: np.ndarray | None = None,
    refit_on_train_val: bool = True,
) -> tuple[np.ndarray, np.ndarray, Any]:
    model = xgb_regressor_cls(random_state=seed, n_jobs=8, verbosity=0, **params)
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float32)
    if stop_rounds > 0:
        try:
            model.set_params(early_stopping_rounds=stop_rounds, eval_metric="mae")
        except Exception:
            pass
        model.fit(X_train, y_train, eval_set=[(X_val, y_val_used)], verbose=False, **fit_kwargs)
        best_it = getattr(model, "best_iteration", None)
        if best_it is not None:
            val_pred = model.predict(X_val, iteration_range=(0, int(best_it) + 1))
            test_pred = model.predict(X_test, iteration_range=(0, int(best_it) + 1))
        else:
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train, **fit_kwargs)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
    test_model = model
    if refit_on_train_val and len(X_val):
        X_refit = np.concatenate([X_train, X_val], axis=0)
        y_refit = np.concatenate([y_train, y_val_used], axis=0)
        refit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            val_weight = np.full(len(X_val), float(np.mean(sample_weight)), dtype=np.float32)
            refit_kwargs["sample_weight"] = np.concatenate([np.asarray(sample_weight, dtype=np.float32), val_weight], axis=0)
        test_model = clone(model)
        try:
            test_model.set_params(early_stopping_rounds=None)
        except Exception:
            pass
        test_model.fit(X_refit, y_refit, **refit_kwargs)
        test_pred = test_model.predict(X_test)
    return val_pred, test_pred, test_model


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
    sample_weight: np.ndarray | None = None,
    refit_on_train_val: bool = True,
) -> tuple[np.ndarray, np.ndarray, Any]:
    cols = [f"f{i}" for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train, columns=cols)
    val_df = pd.DataFrame(X_val, columns=cols)
    test_df = pd.DataFrame(X_test, columns=cols)
    model = lgbm_regressor_cls(random_state=seed, n_jobs=8, verbose=-1, **params)
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float32)
    callbacks = [early_stopping(stopping_rounds=stop_rounds, verbose=False)] if stop_rounds > 0 else None
    if callbacks:
        model.fit(
            train_df,
            y_train,
            eval_set=[(val_df, y_val_used)],
            eval_metric="l1",
            callbacks=callbacks,
            **fit_kwargs,
        )
        num_it = int(getattr(model, "best_iteration_", 0)) or None
        val_pred = model.predict(val_df, num_iteration=num_it)
        test_pred = model.predict(test_df, num_iteration=num_it)
    else:
        model.fit(train_df, y_train, **fit_kwargs)
        val_pred = model.predict(val_df)
        test_pred = model.predict(test_df)
    test_model = model
    if refit_on_train_val and len(X_val):
        X_refit = np.concatenate([X_train, X_val], axis=0)
        y_refit = np.concatenate([y_train, y_val_used], axis=0)
        refit_cols = [f"f{i}" for i in range(X_refit.shape[1])]
        X_refit_df = pd.DataFrame(X_refit, columns=refit_cols)
        refit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            val_weight = np.full(len(X_val), float(np.mean(sample_weight)), dtype=np.float32)
            refit_kwargs["sample_weight"] = np.concatenate([np.asarray(sample_weight, dtype=np.float32), val_weight], axis=0)
        test_model = clone(model)
        test_model.fit(X_refit_df, y_refit, **refit_kwargs)
        test_pred = test_model.predict(test_df)
    return val_pred, test_pred, test_model


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

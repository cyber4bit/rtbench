from __future__ import annotations

from typing import Any

import numpy as np

from ...metrics import compute_metrics
from ..trees import CandidateOutput, _inverse_target
from .common import CandidateBuildContext, predict_lgbm, predict_xgb


def build_hilic_pool_candidates(
    ctx: CandidateBuildContext,
    *,
    xgb_regressor_cls: type[Any],
    lgbm_regressor_cls: type[Any],
) -> list[CandidateOutput]:
    if not bool(ctx.model_cfg.get("ENABLE_HILIC_POOL_CANDIDATES", False)):
        return []
    if ctx.source_row_dataset_ids is None:
        return []

    allowed_ids = _id_set(ctx.model_cfg.get("HILIC_POOL_ALLOWED_SOURCE_IDS"))
    excluded_ids = _id_set(ctx.model_cfg.get("HILIC_POOL_EXCLUDED_SOURCE_IDS"))
    if not allowed_ids:
        return []

    source_ids = np.asarray([str(x).zfill(4) for x in np.asarray(ctx.source_row_dataset_ids, dtype=object)], dtype=object)
    keep = np.asarray([value in allowed_ids for value in source_ids], dtype=bool)
    if excluded_ids:
        keep &= np.asarray([value not in excluded_ids for value in source_ids], dtype=bool)
    min_source_rows = int(ctx.model_cfg.get("HILIC_POOL_MIN_SOURCE_ROWS", 32))
    if int(np.sum(keep)) < min_source_rows:
        return []

    X_src = ctx.X_src[keep]
    y_src = ctx.y_src[keep]
    if ctx.source_sample_weights is not None:
        source_w = np.asarray(ctx.source_sample_weights, dtype=np.float32)[keep]
    else:
        source_w = np.full(len(y_src), float(ctx.source_weight), dtype=np.float32)
    X_train = np.concatenate([X_src, ctx.X_train], axis=0)
    y_train = np.concatenate([y_src, ctx.y_train], axis=0)
    train_w = np.concatenate([source_w, np.full(len(ctx.y_train), float(ctx.target_weight), dtype=np.float32)], axis=0)

    cfg = dict(ctx.model_cfg.get("HILIC_POOL", {}) or {})
    stop_rounds = int(cfg.get("early_stopping_rounds", ctx.model_cfg.get("EARLY_STOPPING_ROUNDS", 0)))
    refit_on_train_val = bool(ctx.model_cfg.get("REFIT_ON_TRAIN_VAL", True))
    candidates: list[CandidateOutput] = []

    if bool(cfg.get("enable_lgbm", True)):
        lgbm_params = dict(
            cfg.get(
                "lgbm",
                {
                    "n_estimators": 900,
                    "num_leaves": 31,
                    "learning_rate": 0.03,
                    "objective": "mae",
                    "feature_fraction": 0.85,
                    "bagging_fraction": 0.9,
                    "bagging_freq": 1,
                    "min_child_samples": 8,
                },
            )
        )
        val_used, test_used, model = predict_lgbm(
            lgbm_regressor_cls,
            params=lgbm_params,
            seed=ctx.seed + 5100,
            X_train=X_train,
            y_train=y_train,
            X_val=ctx.X_val,
            y_val_used=ctx.y_val_used,
            X_test=ctx.X_test,
            stop_rounds=stop_rounds,
            sample_weight=train_w,
            refit_on_train_val=refit_on_train_val,
        )
        val_pred = _inverse_target(val_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
        candidates.append(
            CandidateOutput(
                name="HILIC_POOL_LGBM",
                val_pred=val_pred,
                test_pred=_inverse_target(test_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec),
                val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                model=model,
            )
        )

    if bool(cfg.get("enable_xgb", False)):
        xgb_params = dict(
            cfg.get(
                "xgb",
                {
                    "n_estimators": 800,
                    "max_depth": 5,
                    "learning_rate": 0.03,
                    "objective": "reg:absoluteerror",
                    "subsample": 0.9,
                    "colsample_bytree": 0.85,
                    "reg_lambda": 2.0,
                },
            )
        )
        val_used, test_used, model = predict_xgb(
            xgb_regressor_cls,
            params=xgb_params,
            seed=ctx.seed + 5200,
            X_train=X_train,
            y_train=y_train,
            X_val=ctx.X_val,
            y_val_used=ctx.y_val_used,
            X_test=ctx.X_test,
            stop_rounds=stop_rounds,
            sample_weight=train_w,
            refit_on_train_val=refit_on_train_val,
        )
        val_pred = _inverse_target(val_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
        candidates.append(
            CandidateOutput(
                name="HILIC_POOL_XGB",
                val_pred=val_pred,
                test_pred=_inverse_target(test_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec),
                val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                model=model,
            )
        )
    return candidates


def _id_set(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, (str, bytes)):
        values = [raw]
    else:
        try:
            values = list(raw)
        except TypeError:
            values = [raw]
    return {str(value).zfill(4) for value in values if str(value).strip()}

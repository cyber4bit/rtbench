from __future__ import annotations

from typing import Any

import numpy as np

from ...metrics import compute_metrics
from ..trees import CandidateOutput, _inverse_target, _mdl_feature_subset
from .common import CandidateBuildContext, TreeFitFn, iter_transfer_target_views, predict_lgbm, predict_xgb


def build_mdl_transfer_tree_candidates(
    ctx: CandidateBuildContext,
    *,
    fit_tree_models_fn: TreeFitFn,
) -> list[CandidateOutput]:
    if not bool(ctx.model_cfg.get("ENABLE_MDL_SUBSET_CANDIDATES", False)):
        return []

    X_src_sub, X_train_sub, X_val_sub, X_test_sub = _subset_views(ctx)
    candidates = [
        *fit_tree_models_fn(
            model_cfg=ctx.model_cfg,
            X_src=X_src_sub,
            y_src=ctx.y_src,
            X_t_train=X_train_sub,
            y_t_train=ctx.y_train,
            X_val=X_val_sub,
            y_val_used=ctx.y_val_used,
            y_val_sec=ctx.y_val_sec,
            X_test=X_test_sub,
            seed=ctx.seed + 211,
            source_weight=ctx.source_weight,
            target_weight=ctx.target_weight,
            source_sample_weights=ctx.source_sample_weights,
            name_prefix="MDLFEAT_",
            target_transform=ctx.target_transform,
            target_inv_scale=ctx.target_inv_scale,
            target_t0_sec=ctx.target_t0_sec,
        )
    ]
    for index, tfm_name, y_src_alt, y_train_alt, y_val_alt in iter_transfer_target_views(ctx):
        candidates.extend(
            fit_tree_models_fn(
                model_cfg=ctx.model_cfg,
                X_src=X_src_sub,
                y_src=y_src_alt,
                X_t_train=X_train_sub,
                y_t_train=y_train_alt,
                X_val=X_val_sub,
                y_val_used=y_val_alt,
                y_val_sec=ctx.y_val_sec,
                X_test=X_test_sub,
                seed=ctx.seed + 1500 + index * 37,
                source_weight=ctx.source_weight,
                target_weight=ctx.target_weight,
                source_sample_weights=ctx.source_sample_weights,
                name_prefix=f"XTFM_{tfm_name.upper()}_MDLFEAT_",
                target_transform=tfm_name,
                target_inv_scale=ctx.target_inv_scale,
                target_t0_sec=ctx.target_t0_sec,
            )
        )
    return candidates


def build_mdl_local_tree_candidates(
    ctx: CandidateBuildContext,
    *,
    xgb_regressor_cls: type[Any],
    lgbm_regressor_cls: type[Any],
    local_xgb: dict[str, Any],
    local_lgbm: dict[str, Any],
    stop_rounds: int,
) -> list[CandidateOutput]:
    if not bool(ctx.model_cfg.get("ENABLE_MDL_SUBSET_CANDIDATES", False)):
        return []

    _, X_train_sub, X_val_sub, X_test_sub = _subset_views(ctx)
    candidates: list[CandidateOutput] = []

    val_pred_used, test_pred_used, model = predict_xgb(
        xgb_regressor_cls,
        params=local_xgb,
        seed=ctx.seed + 650,
        X_train=X_train_sub,
        y_train=ctx.y_train,
        X_val=X_val_sub,
        y_val_used=ctx.y_val_used,
        X_test=X_test_sub,
        stop_rounds=stop_rounds,
    )
    val_pred = _inverse_target(val_pred_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    candidates.append(
        CandidateOutput(
            name="LOCAL_XGB_MDLFEAT",
            val_pred=val_pred,
            test_pred=_inverse_target(test_pred_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec),
            val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
            model=model,
        )
    )

    val_pred_used, test_pred_used, model = predict_lgbm(
        lgbm_regressor_cls,
        params=local_lgbm,
        seed=ctx.seed + 660,
        X_train=X_train_sub,
        y_train=ctx.y_train,
        X_val=X_val_sub,
        y_val_used=ctx.y_val_used,
        X_test=X_test_sub,
        stop_rounds=stop_rounds,
    )
    val_pred = _inverse_target(val_pred_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    candidates.append(
        CandidateOutput(
            name="LOCAL_LGBM_MDLFEAT",
            val_pred=val_pred,
            test_pred=_inverse_target(test_pred_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec),
            val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
            model=model,
        )
    )
    return candidates


def _subset_views(ctx: CandidateBuildContext) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        _mdl_feature_subset(ctx.X_src, ctx.group_sizes).astype(np.float32),
        _mdl_feature_subset(ctx.X_train, ctx.group_sizes).astype(np.float32),
        _mdl_feature_subset(ctx.X_val, ctx.group_sizes).astype(np.float32),
        _mdl_feature_subset(ctx.X_test, ctx.group_sizes).astype(np.float32),
    )

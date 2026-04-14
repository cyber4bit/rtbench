from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...metrics import compute_metrics
from ..trees import CandidateOutput, _fit_branch_tree_models, _inverse_target
from .common import CandidateBuildContext, TreeFitFn, iter_local_target_views, iter_transfer_target_views, predict_lgbm, predict_xgb


@dataclass(frozen=True)
class LocalTreeSettings:
    xgb: dict[str, Any]
    lgbm: dict[str, Any]
    stop_rounds: int


def resolve_local_tree_settings(ctx: CandidateBuildContext) -> LocalTreeSettings:
    return LocalTreeSettings(
        xgb=ctx.model_cfg.get(
            "LOCAL_XGB",
            {
                "n_estimators": 1400,
                "max_depth": 6,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "objective": "reg:absoluteerror",
                "reg_lambda": 2.0,
            },
        ),
        lgbm=ctx.model_cfg.get(
            "LOCAL_LGBM",
            {
                "n_estimators": 2200,
                "num_leaves": 63,
                "learning_rate": 0.03,
                "objective": "mae",
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
            },
        ),
        stop_rounds=int(ctx.model_cfg.get("EARLY_STOPPING_ROUNDS", 0)),
    )


def build_transfer_tree_candidates(
    ctx: CandidateBuildContext,
    *,
    fit_tree_models_fn: TreeFitFn,
) -> list[CandidateOutput]:
    candidates = [
        *fit_tree_models_fn(
            model_cfg=ctx.model_cfg,
            X_src=ctx.X_src,
            y_src=ctx.y_src,
            X_t_train=ctx.X_train,
            y_t_train=ctx.y_train,
            X_val=ctx.X_val,
            y_val_used=ctx.y_val_used,
            y_val_sec=ctx.y_val_sec,
            X_test=ctx.X_test,
            seed=ctx.seed,
            source_weight=ctx.source_weight,
            target_weight=ctx.target_weight,
            source_sample_weights=ctx.source_sample_weights,
            target_transform=ctx.target_transform,
            target_inv_scale=ctx.target_inv_scale,
            target_t0_sec=ctx.target_t0_sec,
        )
    ]
    for index, tfm_name, y_src_alt, y_train_alt, y_val_alt in iter_transfer_target_views(ctx):
        candidates.extend(
            fit_tree_models_fn(
                model_cfg=ctx.model_cfg,
                X_src=ctx.X_src,
                y_src=y_src_alt,
                X_t_train=ctx.X_train,
                y_t_train=y_train_alt,
                X_val=ctx.X_val,
                y_val_used=y_val_alt,
                y_val_sec=ctx.y_val_sec,
                X_test=ctx.X_test,
                seed=ctx.seed + 1200 + index * 37,
                source_weight=ctx.source_weight,
                target_weight=ctx.target_weight,
                source_sample_weights=ctx.source_sample_weights,
                name_prefix=f"XTFM_{tfm_name.upper()}_",
                target_transform=tfm_name,
                target_inv_scale=ctx.target_inv_scale,
                target_t0_sec=ctx.target_t0_sec,
            )
        )
    return candidates


def build_fail_tune_candidates(ctx: CandidateBuildContext) -> list[CandidateOutput]:
    return _fit_branch_tree_models(
        branch_name="FULL",
        specs=[
            (
                "XGB_FAIL",
                "xgb",
                ctx.model_cfg.get(
                    "FAIL_XGB",
                    {
                        "n_estimators": 3600,
                        "max_depth": 7,
                        "learning_rate": 0.015,
                        "subsample": 0.95,
                        "colsample_bytree": 0.8,
                        "objective": "reg:absoluteerror",
                        "reg_lambda": 3.0,
                    },
                ),
            ),
            (
                "LGBM_FAIL",
                "lgbm",
                ctx.model_cfg.get(
                    "FAIL_LGBM",
                    {
                        "n_estimators": 5000,
                        "num_leaves": 127,
                        "learning_rate": 0.015,
                        "objective": "mae",
                        "feature_fraction": 0.85,
                        "bagging_fraction": 0.9,
                        "bagging_freq": 1,
                    },
                ),
            ),
        ],
        X_src=ctx.X_src,
        y_src=ctx.y_src,
        X_t_train=ctx.X_train,
        y_t_train=ctx.y_train,
        X_val=ctx.X_val,
        y_val_used=ctx.y_val_used,
        y_val_sec=ctx.y_val_sec,
        X_test=ctx.X_test,
        seed=ctx.seed + 300,
        source_weight=ctx.source_weight,
        target_weight=ctx.target_weight,
        early_stopping_rounds=int(ctx.model_cfg.get("EARLY_STOPPING_ROUNDS", 0)),
        source_sample_weights=ctx.source_sample_weights,
        target_transform=ctx.target_transform,
        target_inv_scale=ctx.target_inv_scale,
        target_t0_sec=ctx.target_t0_sec,
    )


def build_local_tree_candidates(
    ctx: CandidateBuildContext,
    *,
    xgb_regressor_cls: type[Any],
    lgbm_regressor_cls: type[Any],
    settings: LocalTreeSettings,
) -> list[CandidateOutput]:
    candidates = [
        _build_single_local_candidate(
            ctx,
            name="LOCAL_XGB",
            regressor_cls=xgb_regressor_cls,
            params=settings.xgb,
            seed=ctx.seed + 500,
            stop_rounds=settings.stop_rounds,
            use_lgbm=False,
        ),
        _build_single_local_candidate(
            ctx,
            name="LOCAL_LGBM",
            regressor_cls=lgbm_regressor_cls,
            params=settings.lgbm,
            seed=ctx.seed + 600,
            stop_rounds=settings.stop_rounds,
            use_lgbm=True,
        ),
    ]
    for index, tfm_name, y_train_alt, y_val_alt in iter_local_target_views(ctx):
        val_pred_used, test_pred_used, model = predict_xgb(
            xgb_regressor_cls,
            params=settings.xgb,
            seed=ctx.seed + 800 + index * 17,
            X_train=ctx.X_train,
            y_train=y_train_alt,
            X_val=ctx.X_val,
            y_val_used=y_val_alt,
            X_test=ctx.X_test,
            stop_rounds=settings.stop_rounds,
        )
        val_pred = _inverse_target(val_pred_used, tfm_name, ctx.target_inv_scale, ctx.target_t0_sec)
        candidates.append(
            CandidateOutput(
                name=f"LOCAL_XGB_{tfm_name.upper()}",
                val_pred=val_pred,
                test_pred=_inverse_target(test_pred_used, tfm_name, ctx.target_inv_scale, ctx.target_t0_sec),
                val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                model=model,
            )
        )
    return candidates


def _build_single_local_candidate(
    ctx: CandidateBuildContext,
    *,
    name: str,
    regressor_cls: type[Any],
    params: dict[str, Any],
    seed: int,
    stop_rounds: int,
    use_lgbm: bool,
) -> CandidateOutput:
    predictor = predict_lgbm if use_lgbm else predict_xgb
    val_pred_used, test_pred_used, model = predictor(
        regressor_cls,
        params=params,
        seed=seed,
        X_train=ctx.X_train,
        y_train=ctx.y_train,
        X_val=ctx.X_val,
        y_val_used=ctx.y_val_used,
        X_test=ctx.X_test,
        stop_rounds=stop_rounds,
    )
    val_pred = _inverse_target(val_pred_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    return CandidateOutput(
        name=name,
        val_pred=val_pred,
        test_pred=_inverse_target(test_pred_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec),
        val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
        model=model,
    )

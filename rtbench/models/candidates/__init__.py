from __future__ import annotations

import re

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from ..mlp import train_mlp as _train_mlp
from ..ridge import _fit_ridge_models
from ..trees import CandidateOutput, _fit_tree_models
from .anchor import build_anchor_candidates, build_calibrated_mol_lookup_candidates, build_exact_mol_lookup_candidates
from .common import CandidateBuildContext, MLPTrainFn, RidgeFitFn, TreeFitFn
from .hilic_pool import build_hilic_pool_candidates
from .hyper_candidates import build_hyper_candidates
from .local_fast import build_local_fast_candidates
from .mdl_subset import build_mdl_local_tree_candidates, build_mdl_transfer_tree_candidates
from .ridge_candidates import build_ridge_candidates
from .tree_candidates import build_fail_tune_candidates, build_local_tree_candidates, build_transfer_tree_candidates, resolve_local_tree_settings


def _normalize_patterns(values: object) -> list[re.Pattern[str]]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raw_values = [values]
    else:
        try:
            raw_values = list(values)
        except TypeError:
            raw_values = [values]
    patterns: list[re.Pattern[str]] = []
    for value in raw_values:
        text = str(value).strip()
        if text:
            patterns.append(re.compile(text))
    return patterns


def _filter_candidate_names(candidates: list[CandidateOutput], model_cfg: dict[str, object]) -> list[CandidateOutput]:
    allowlist = _normalize_patterns(model_cfg.get("CANDIDATE_NAME_ALLOWLIST"))
    denylist = _normalize_patterns(model_cfg.get("CANDIDATE_NAME_DENYLIST"))
    filtered = candidates
    if allowlist:
        filtered = [candidate for candidate in filtered if any(pattern.search(candidate.name) for pattern in allowlist)]
    if denylist:
        filtered = [candidate for candidate in filtered if not any(pattern.search(candidate.name) for pattern in denylist)]
    return filtered


def build_candidates(
    ctx: CandidateBuildContext,
    *,
    fit_tree_models_fn: TreeFitFn = _fit_tree_models,
    fit_ridge_models_fn: RidgeFitFn = _fit_ridge_models,
    train_mlp_fn: MLPTrainFn = _train_mlp,
    xgb_regressor_cls: type[object] = XGBRegressor,
    lgbm_regressor_cls: type[object] = LGBMRegressor,
) -> list[CandidateOutput]:
    target_rows = int(len(ctx.X_train)) + int(len(ctx.X_val)) + int(len(ctx.X_test))
    if target_rows <= 0:
        raise ValueError("Empty target dataset: no target samples available for candidate generation.")

    candidates = build_hyper_candidates(ctx)
    candidates.extend(build_exact_mol_lookup_candidates(ctx))
    candidates.extend(build_calibrated_mol_lookup_candidates(ctx))
    candidates.extend(
        build_hilic_pool_candidates(
            ctx,
            xgb_regressor_cls=xgb_regressor_cls,
            lgbm_regressor_cls=lgbm_regressor_cls,
        )
    )
    candidates.extend(build_local_fast_candidates(ctx))
    if bool(ctx.model_cfg.get("ONLY_HYPER_TL", False)) or bool(ctx.model_cfg.get("EARLY_CANDIDATES_ONLY", False)):
        candidates = _filter_candidate_names(candidates, ctx.model_cfg)
        if not candidates:
            raise ValueError("No early-stage transfer or lookup candidates were produced.")
        return candidates

    if not bool(ctx.model_cfg.get("DISABLE_TRANSFER_TREE_CANDIDATES", False)):
        candidates.extend(build_transfer_tree_candidates(ctx, fit_tree_models_fn=fit_tree_models_fn))
    if not bool(ctx.model_cfg.get("DISABLE_RIDGE_CANDIDATES", False)):
        candidates.extend(build_ridge_candidates(ctx, fit_ridge_models_fn=fit_ridge_models_fn))
    if not bool(ctx.model_cfg.get("DISABLE_MDL_TRANSFER_TREE_CANDIDATES", False)):
        candidates.extend(build_mdl_transfer_tree_candidates(ctx, fit_tree_models_fn=fit_tree_models_fn))
    if bool(ctx.model_cfg.get("ENABLE_MLP", True)):
        candidates.append(
            train_mlp_fn(
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
                target_transform=ctx.target_transform,
                target_inv_scale=ctx.target_inv_scale,
                target_t0_sec=ctx.target_t0_sec,
            )
        )
    if ctx.fail_tune:
        candidates.extend(build_fail_tune_candidates(ctx))

    local_settings = resolve_local_tree_settings(ctx)
    if not bool(ctx.model_cfg.get("DISABLE_LOCAL_TREE_CANDIDATES", False)):
        candidates.extend(
            build_local_tree_candidates(
                ctx,
                xgb_regressor_cls=xgb_regressor_cls,
                lgbm_regressor_cls=lgbm_regressor_cls,
                settings=local_settings,
            )
        )
    if not bool(ctx.model_cfg.get("DISABLE_MDL_LOCAL_TREE_CANDIDATES", False)):
        candidates.extend(
            build_mdl_local_tree_candidates(
                ctx,
                xgb_regressor_cls=xgb_regressor_cls,
                lgbm_regressor_cls=lgbm_regressor_cls,
                local_xgb=local_settings.xgb,
                local_lgbm=local_settings.lgbm,
                stop_rounds=local_settings.stop_rounds,
            )
        )
    if not bool(ctx.model_cfg.get("DISABLE_ANCHOR_CANDIDATES", False)):
        candidates.extend(
            build_anchor_candidates(
                ctx,
                xgb_regressor_cls=xgb_regressor_cls,
                lgbm_regressor_cls=lgbm_regressor_cls,
                local_xgb=local_settings.xgb,
                local_lgbm=local_settings.lgbm,
                stop_rounds=local_settings.stop_rounds,
            )
        )
    candidates = _filter_candidate_names(candidates, ctx.model_cfg)
    if not candidates:
        raise ValueError("No ensemble candidates were produced.")
    return candidates


collect_candidates = build_candidates

__all__ = ["CandidateBuildContext", "build_candidates", "collect_candidates"]

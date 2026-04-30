from __future__ import annotations

from ..trees import CandidateOutput
from .common import CandidateBuildContext, RidgeFitFn, iter_transfer_target_views


def build_ridge_candidates(
    ctx: CandidateBuildContext,
    *,
    fit_ridge_models_fn: RidgeFitFn,
) -> list[CandidateOutput]:
    candidates = [
        *fit_ridge_models_fn(
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
    for index, tfm_name, y_src_alt, y_train_alt, _y_val_alt in iter_transfer_target_views(ctx):
        candidates.extend(
            fit_ridge_models_fn(
                model_cfg=ctx.model_cfg,
                X_src=ctx.X_src,
                y_src=y_src_alt,
                X_t_train=ctx.X_train,
                y_t_train=y_train_alt,
                X_val=ctx.X_val,
                y_val_used=_y_val_alt,
                y_val_sec=ctx.y_val_sec,
                X_test=ctx.X_test,
                seed=ctx.seed + 1300 + index * 37,
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

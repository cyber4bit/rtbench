from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ...metrics import compute_metrics
from ..calibration import apply_calibration, calibrate_linear
from ..trees import CandidateOutput, _forward_target, _inverse_target
from .common import CandidateBuildContext, clean_mol_key, cosine, predict_lgbm, predict_xgb


def build_anchor_candidates(
    ctx: CandidateBuildContext,
    *,
    xgb_regressor_cls: type[Any],
    lgbm_regressor_cls: type[Any],
    local_xgb: dict[str, Any],
    local_lgbm: dict[str, Any],
    stop_rounds: int,
) -> list[CandidateOutput]:
    enabled = bool(ctx.model_cfg.get("ENABLE_ANCHOR_TL", False))
    if not enabled or ctx.source_mol_keys is None or ctx.target_mol_keys is None or ctx.target_cp_reference is None:
        return []

    source_ds_ids = (
        np.asarray(ctx.source_row_dataset_ids, dtype=object)
        if ctx.source_row_dataset_ids is not None
        else np.array(["src"] * len(ctx.source_mol_keys), dtype=object)
    )
    prior_sec_all = _build_anchor_prior_sec(
        source_mol_keys=np.asarray(ctx.source_mol_keys, dtype=object),
        source_row_dataset_ids=source_ds_ids,
        y_src_sec_raw=np.asarray(ctx.y_src_sec, dtype=np.float32),
        source_ds_cp=_source_cp_by_dataset(ctx, source_ds_ids),
        target_cp_vec=np.asarray(ctx.target_cp_reference, dtype=np.float32),
        target_mol_keys=np.asarray(ctx.target_mol_keys, dtype=object),
        anchor_cfg=dict(ctx.model_cfg.get("ANCHOR_TL", {}) or {}),
    )
    return _build_anchor_outputs(
        ctx,
        prior_sec_all=prior_sec_all,
        xgb_regressor_cls=xgb_regressor_cls,
        lgbm_regressor_cls=lgbm_regressor_cls,
        local_xgb=local_xgb,
        local_lgbm=local_lgbm,
        stop_rounds=stop_rounds,
    )


def _build_anchor_outputs(
    ctx: CandidateBuildContext,
    *,
    prior_sec_all: np.ndarray,
    xgb_regressor_cls: type[Any],
    lgbm_regressor_cls: type[Any],
    local_xgb: dict[str, Any],
    local_lgbm: dict[str, Any],
    stop_rounds: int,
) -> list[CandidateOutput]:
    anchor_cfg = dict(ctx.model_cfg.get("ANCHOR_TL", {}) or {})
    train_size = len(ctx.X_train)
    val_size = len(ctx.X_val)
    p_tr_sec = prior_sec_all[:train_size]
    p_va_sec = prior_sec_all[train_size : train_size + val_size]
    p_te_sec = prior_sec_all[train_size + val_size :]
    has_tr = np.isfinite(p_tr_sec)
    has_va = np.isfinite(p_va_sec)
    has_te = np.isfinite(p_te_sec)
    if int(np.sum(has_tr)) < int(anchor_cfg.get("min_train_points", 6)):
        return []

    a, b = calibrate_linear(ctx.y_train_sec[has_tr], p_tr_sec[has_tr])
    fill_sec = float(np.median(ctx.y_train_sec))
    candidates = [_build_anchor_linear_candidate(ctx, p_va_sec, p_te_sec, has_va, has_te, a, b, fill_sec)]
    Xtr_a, Xva_a, Xte_a = _anchor_feature_views(ctx, p_tr_sec, p_va_sec, p_te_sec, has_tr, has_va, has_te, fill_sec)

    val_pred_used, test_pred_used, model = predict_xgb(
        xgb_regressor_cls,
        params=ctx.model_cfg.get("ANCHOR_LOCAL_XGB", local_xgb),
        seed=ctx.seed + 930,
        X_train=Xtr_a,
        y_train=ctx.y_train,
        X_val=Xva_a,
        y_val_used=ctx.y_val_used,
        X_test=Xte_a,
        stop_rounds=stop_rounds,
    )
    val_pred = _inverse_target(val_pred_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    candidates.append(
        CandidateOutput(
            name="LOCAL_ANCHOR_XGB",
            val_pred=val_pred,
            test_pred=_inverse_target(test_pred_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec),
            val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
            model=model,
        )
    )

    val_pred_used, test_pred_used, model = predict_lgbm(
        lgbm_regressor_cls,
        params=ctx.model_cfg.get("ANCHOR_LOCAL_LGBM", local_lgbm),
        seed=ctx.seed + 940,
        X_train=Xtr_a,
        y_train=ctx.y_train,
        X_val=Xva_a,
        y_val_used=ctx.y_val_used,
        X_test=Xte_a,
        stop_rounds=stop_rounds,
    )
    val_pred = _inverse_target(val_pred_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    candidates.append(
        CandidateOutput(
            name="LOCAL_ANCHOR_LGBM",
            val_pred=val_pred,
            test_pred=_inverse_target(test_pred_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec),
            val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
            model=model,
        )
    )
    return candidates


def _build_anchor_linear_candidate(
    ctx: CandidateBuildContext,
    p_va_sec: np.ndarray,
    p_te_sec: np.ndarray,
    has_va: np.ndarray,
    has_te: np.ndarray,
    a: float,
    b: float,
    fill_sec: float,
) -> CandidateOutput:
    val_pred = np.full(len(p_va_sec), fill_sec, dtype=np.float32)
    test_pred = np.full(len(p_te_sec), fill_sec, dtype=np.float32)
    if np.any(has_va):
        val_pred[has_va] = apply_calibration(a, b, p_va_sec[has_va])
    if np.any(has_te):
        test_pred[has_te] = apply_calibration(a, b, p_te_sec[has_te])
    return CandidateOutput(
        name="ANCHOR_LINEAR",
        val_pred=val_pred,
        test_pred=test_pred,
        val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
        model={"type": "anchor_linear"},
    )


def _anchor_feature_views(
    ctx: CandidateBuildContext,
    p_tr_sec: np.ndarray,
    p_va_sec: np.ndarray,
    p_te_sec: np.ndarray,
    has_tr: np.ndarray,
    has_va: np.ndarray,
    has_te: np.ndarray,
    fill_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_tr_used = _forward_target(np.where(has_tr, p_tr_sec, fill_sec), ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    p_va_used = _forward_target(np.where(has_va, p_va_sec, fill_sec), ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    p_te_used = _forward_target(np.where(has_te, p_te_sec, fill_sec), ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    return (
        np.concatenate([ctx.X_train, p_tr_used.reshape(-1, 1), has_tr.astype(np.float32).reshape(-1, 1)], axis=1).astype(np.float32),
        np.concatenate([ctx.X_val, p_va_used.reshape(-1, 1), has_va.astype(np.float32).reshape(-1, 1)], axis=1).astype(np.float32),
        np.concatenate([ctx.X_test, p_te_used.reshape(-1, 1), has_te.astype(np.float32).reshape(-1, 1)], axis=1).astype(np.float32),
    )


def _build_anchor_prior_sec(
    source_mol_keys: np.ndarray,
    source_row_dataset_ids: np.ndarray,
    y_src_sec_raw: np.ndarray,
    source_ds_cp: dict[str, np.ndarray],
    target_cp_vec: np.ndarray,
    target_mol_keys: np.ndarray,
    anchor_cfg: dict[str, Any],
) -> np.ndarray:
    sim_power = float(anchor_cfg.get("similarity_power", 2.0))
    min_scale = float(anchor_cfg.get("min_scale", 0.25))
    max_scale = float(anchor_cfg.get("max_scale", 2.0))
    top_k_sources = anchor_cfg.get("top_k_sources", None)
    try:
        top_k_sources = int(top_k_sources) if top_k_sources is not None else None
    except Exception:
        top_k_sources = None

    ds_sims = sorted(
        ((dataset, max(0.0, cosine(target_cp_vec, cp_vec))) for dataset, cp_vec in source_ds_cp.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ds_sims:
        return np.full(len(target_mol_keys), np.nan, dtype=np.float32)
    if top_k_sources is not None and top_k_sources > 0:
        ds_sims = ds_sims[: min(top_k_sources, len(ds_sims))]
    sim_vals = np.array([item[1] for item in ds_sims], dtype=np.float64)
    if np.allclose(sim_vals, 0.0):
        sim_vals = np.ones_like(sim_vals)
    sim_vals = np.clip(np.power(sim_vals, sim_power) / max(float(np.mean(sim_vals)), 1e-12), min_scale, max_scale)
    dataset_weights = {dataset: float(sim_vals[i]) for i, (dataset, _sim) in enumerate(ds_sims)}

    sum_w = defaultdict(float)
    sum_y = defaultdict(float)
    for mol_key, dataset_id, value in zip(source_mol_keys, source_row_dataset_ids, y_src_sec_raw):
        key = clean_mol_key(mol_key)
        if not key or not np.isfinite(float(value)):
            continue
        weight = dataset_weights.get(str(dataset_id), 0.0)
        if weight <= 0.0:
            continue
        sum_w[key] += weight
        sum_y[key] += weight * float(value)

    out = np.full(len(target_mol_keys), np.nan, dtype=np.float32)
    for index, mol_key in enumerate(target_mol_keys):
        key = clean_mol_key(mol_key)
        total_weight = sum_w.get(key, 0.0)
        if total_weight > 0.0:
            out[index] = float(sum_y[key] / total_weight)
    return out


def _source_cp_by_dataset(ctx: CandidateBuildContext, source_ds_ids: np.ndarray) -> dict[str, np.ndarray]:
    if ctx.source_row_dataset_ids is None:
        return {"src": np.asarray(ctx.X_src_cp[0], dtype=np.float32)}
    out: dict[str, np.ndarray] = {}
    for index, dataset_id in enumerate(source_ds_ids.tolist()):
        key = str(dataset_id)
        if key not in out:
            out[key] = np.asarray(ctx.X_src_cp[index], dtype=np.float32)
    return out

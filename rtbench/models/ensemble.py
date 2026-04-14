from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..hyper import HyperTLBundle
from ..metrics import compute_metrics
from .calibration import (
    apply_calibration as _apply_calibration,
    calibrate_candidates,
    calibrate_linear as _calibrate_linear,
    optimize_weights as _optimize_weights,
)
from .candidates import CandidateBuildContext, build_candidates
from .importance import feature_group_importance as _feature_group_importance
from .trees import SplitData, _inverse_target


@dataclass
class EnsembleOutput:
    pred_test: np.ndarray
    pred_val: np.ndarray
    top_models: list[str]
    weights: list[float]
    feature_group_importance: dict[str, float]


def _build_candidate_context(
    *,
    model_cfg: dict[str, Any],
    X_src: np.ndarray,
    X_src_cp: np.ndarray,
    y_src: np.ndarray,
    X_target: np.ndarray,
    X_target_mol: np.ndarray,
    X_target_cp: np.ndarray,
    y_target: np.ndarray,
    split: SplitData,
    seed: int,
    source_weight: float,
    target_weight: float,
    group_sizes: dict[str, int],
    y_target_sec: np.ndarray | None,
    y_src_sec_raw: np.ndarray | None,
    fail_tune: bool,
    source_sample_weights: np.ndarray | None,
    target_transform: str,
    target_inv_scale: float,
    target_t0_sec: float,
    hyper_bundle: HyperTLBundle | list[HyperTLBundle] | None,
    source_row_dataset_ids: np.ndarray | None,
    source_mol_keys: np.ndarray | None,
    target_mol_keys: np.ndarray | None,
) -> CandidateBuildContext:
    y_train = y_target[split.train_idx]
    y_val_used = y_target[split.val_idx]
    y_val_sec = (
        np.asarray(y_target_sec, dtype=np.float32)[split.val_idx]
        if y_target_sec is not None
        else _inverse_target(y_val_used, target_transform, target_inv_scale, target_t0_sec)
    )
    y_train_sec = (
        np.asarray(y_target_sec, dtype=np.float32)[split.train_idx]
        if y_target_sec is not None
        else _inverse_target(y_train, target_transform, target_inv_scale, target_t0_sec)
    )
    y_src_sec = (
        np.asarray(y_src_sec_raw, dtype=np.float32)
        if y_src_sec_raw is not None
        else _inverse_target(y_src, target_transform, target_inv_scale, target_t0_sec)
    )
    target_keys = None
    if target_mol_keys is not None:
        ordered_idx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
        target_keys = np.asarray(target_mol_keys, dtype=object)[ordered_idx]
    return CandidateBuildContext(
        model_cfg=model_cfg,
        X_src=X_src,
        X_src_cp=X_src_cp,
        y_src=y_src,
        y_src_sec=y_src_sec,
        X_train=X_target[split.train_idx],
        X_train_mol=X_target_mol[split.train_idx],
        X_train_cp=X_target_cp[split.train_idx],
        y_train=y_train,
        y_train_sec=y_train_sec,
        X_val=X_target[split.val_idx],
        X_val_mol=X_target_mol[split.val_idx],
        y_val_used=y_val_used,
        y_val_sec=y_val_sec,
        X_test=X_target[split.test_idx],
        X_test_mol=X_target_mol[split.test_idx],
        seed=seed,
        source_weight=source_weight,
        target_weight=target_weight,
        group_sizes=group_sizes,
        fail_tune=fail_tune,
        source_sample_weights=source_sample_weights,
        target_transform=target_transform,
        target_inv_scale=target_inv_scale,
        target_t0_sec=target_t0_sec,
        hyper_bundle=hyper_bundle,
        source_row_dataset_ids=source_row_dataset_ids,
        source_mol_keys=source_mol_keys,
        target_mol_keys=target_keys,
        target_cp_reference=np.asarray(X_target_cp[0], dtype=np.float32) if len(X_target_cp) else None,
    )

def _sort_key(candidate: Any, rank_mode: str) -> tuple[float, float]:
    mae = float(candidate.val_metrics.get("mae", float("inf")))
    if rank_mode == "mae_then_r2":
        return mae, -float(candidate.val_metrics.get("r2", float("-inf")))
    return mae, 0.0


def _is_valid_candidate(candidate: Any, expected_val_len: int) -> bool:
    try:
        val_pred = np.asarray(candidate.val_pred, dtype=np.float64).reshape(-1)
        test_pred = np.asarray(candidate.test_pred, dtype=np.float64).reshape(-1)
    except Exception:
        return False
    if len(val_pred) != int(expected_val_len):
        return False
    if not np.all(np.isfinite(val_pred)) or not np.all(np.isfinite(test_pred)):
        return False
    mae = float(candidate.val_metrics.get("mae", float("nan")))
    return bool(np.isfinite(mae))


def train_and_ensemble(
    model_cfg: dict[str, dict[str, Any]],
    X_src: np.ndarray,
    X_src_mol: np.ndarray,
    X_src_cp: np.ndarray,
    y_src: np.ndarray,
    X_target: np.ndarray,
    X_target_mol: np.ndarray,
    X_target_cp: np.ndarray,
    y_target: np.ndarray,
    split: SplitData,
    seed: int,
    source_weight: float,
    target_weight: float,
    group_sizes: dict[str, int],
    y_target_sec: np.ndarray | None = None,
    y_src_sec_raw: np.ndarray | None = None,
    fail_tune: bool = False,
    source_sample_weights: np.ndarray | None = None,
    target_transform: str = "none",
    target_inv_scale: float = 1.0,
    target_t0_sec: float = 1.0,
    hyper_bundle: HyperTLBundle | list[HyperTLBundle] | None = None,
    source_row_dataset_ids: np.ndarray | None = None,
    source_mol_keys: np.ndarray | None = None,
    target_mol_keys: np.ndarray | None = None,
) -> EnsembleOutput:
    _ = X_src_mol
    ctx = _build_candidate_context(
        model_cfg=model_cfg,
        X_src=X_src,
        X_src_cp=X_src_cp,
        y_src=y_src,
        X_target=X_target,
        X_target_mol=X_target_mol,
        X_target_cp=X_target_cp,
        y_target=y_target,
        split=split,
        seed=seed,
        source_weight=source_weight,
        target_weight=target_weight,
        group_sizes=group_sizes,
        y_target_sec=y_target_sec,
        y_src_sec_raw=y_src_sec_raw,
        fail_tune=fail_tune,
        source_sample_weights=source_sample_weights,
        target_transform=target_transform,
        target_inv_scale=target_inv_scale,
        target_t0_sec=target_t0_sec,
        hyper_bundle=hyper_bundle,
        source_row_dataset_ids=source_row_dataset_ids,
        source_mol_keys=source_mol_keys,
        target_mol_keys=target_mol_keys,
    )
    candidates = sorted(
        build_candidates(ctx),
        key=lambda candidate: _sort_key(candidate, str(model_cfg.get("FUSION_RANK", "mae")).strip().lower()),
    )

    if bool(model_cfg.get("CALIBRATE", True)):
        calibrate_candidates(candidates, ctx.y_val_sec)

    clip_mult = float(model_cfg.get("CLIP_MULT", 1.5))
    y_clip_max = float(np.nanmax(ctx.y_train_sec)) * clip_mult if len(ctx.y_train_sec) else float("inf")
    if np.isfinite(y_clip_max) and y_clip_max > 0:
        for candidate in candidates:
            candidate.val_pred = np.clip(candidate.val_pred, 0.0, y_clip_max)
            candidate.test_pred = np.clip(candidate.test_pred, 0.0, y_clip_max)
            candidate.val_metrics = compute_metrics(ctx.y_val_sec, candidate.val_pred)
    rank_mode = str(model_cfg.get("FUSION_RANK", "mae")).strip().lower()
    candidates = sorted(candidates, key=lambda candidate: _sort_key(candidate, rank_mode))
    candidates = [candidate for candidate in candidates if _is_valid_candidate(candidate, expected_val_len=len(ctx.y_val_sec))]
    if not candidates:
        raise ValueError("No valid ensemble candidates were produced.")

    top = candidates[: int(model_cfg.get("FUSION_TOP_K", 3))]
    if not top:
        raise ValueError("No ensemble candidates available after ranking.")
    val_mat = np.column_stack([candidate.val_pred for candidate in top])
    test_mat = np.column_stack([candidate.test_pred for candidate in top])
    weights = _optimize_weights(
        ctx.y_val_sec,
        val_mat,
        objective=str(model_cfg.get("FUSION_OBJECTIVE", "mae")),
        r2_weight=float(model_cfg.get("FUSION_R2_WEIGHT", 0.25)),
        l2_reg=float(model_cfg.get("FUSION_L2_REG", 0.0)),
        weight_floor=float(model_cfg.get("FUSION_WEIGHT_FLOOR", 0.0)),
    )
    return EnsembleOutput(
        pred_test=test_mat @ weights,
        pred_val=val_mat @ weights,
        top_models=[candidate.name for candidate in top],
        weights=[float(weight) for weight in weights],
        feature_group_importance=_feature_group_importance(top[0].model, group_sizes),
    )

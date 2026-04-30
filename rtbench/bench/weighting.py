from __future__ import annotations

from typing import Any

import numpy as np


def expand_per_dataset_multiplier(dataset_id: str | int, mapping: dict[str, float] | None, default: float = 1.0) -> float:
    ds = str(dataset_id).zfill(4)
    if not mapping:
        return float(default)
    normalized = {str(k).zfill(4): float(v) for k, v in dict(mapping).items()}
    return float(normalized.get(ds, default))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def build_adaptive_source_weights(
    pretrain_ids: list[str],
    mats: dict[str, Any],
    source_row_dataset_ids: np.ndarray,
    target_ds: str,
    base_source_weight: float,
    similarity_power: float,
    min_scale: float,
    max_scale: float,
    mode: str = "per_sample",
    top_k_sources: int | None = None,
) -> np.ndarray:
    tgt = np.asarray(mats[target_ds].X_cp[0], dtype=np.float64)
    sims = []
    for ds in pretrain_ids:
        src_cp = np.asarray(mats[ds].X_cp[0], dtype=np.float64)
        sim = _cosine(tgt, src_cp)
        sims.append(max(0.0, sim))
    sims_arr = np.asarray(sims, dtype=np.float64)
    sims_raw = sims_arr.copy()
    if np.allclose(sims_arr, 0.0):
        sims_arr = np.ones_like(sims_arr)
    sims_arr = np.power(sims_arr, similarity_power)
    sims_arr = sims_arr / max(np.mean(sims_arr), 1e-12)
    sims_arr = np.clip(sims_arr, min_scale, max_scale)
    if top_k_sources is not None and int(top_k_sources) > 0 and int(top_k_sources) < len(pretrain_ids):
        k = int(top_k_sources)
        top_idx = np.argsort(-sims_raw)[:k]
        keep = set(int(i) for i in top_idx.tolist())
        sims_arr = np.array([sims_arr[i] if i in keep else 0.0 for i in range(len(sims_arr))], dtype=np.float64)
        if np.allclose(sims_arr, 0.0):
            sims_arr = np.ones_like(sims_arr)
    ds_to_scale = {ds: float(sims_arr[i]) for i, ds in enumerate(pretrain_ids)}
    mode = str(mode or "per_sample").strip().lower()
    if mode == "per_dataset":
        ds_to_n = {ds: int(len(mats[ds].y_sec)) for ds in pretrain_ids}
        out = np.array(
            [
                base_source_weight * ds_to_scale[str(ds)] / max(ds_to_n.get(str(ds), 1), 1)
                for ds in source_row_dataset_ids
            ],
            dtype=np.float32,
        )
    else:
        out = np.array([base_source_weight * ds_to_scale[str(ds)] for ds in source_row_dataset_ids], dtype=np.float32)
    return out


def normalize_target_transform(transfer_weights: dict[str, Any]) -> str:
    target_normalize = bool(transfer_weights.get("target_normalize", False))
    target_transform = str(transfer_weights.get("target_transform", "")).strip().lower()
    if not target_transform:
        target_transform = "gradient_norm" if target_normalize else "none"
    if target_transform in ("gradient_norm", "gradnorm", "gradient_normalize", "target_normalize", "normalize"):
        return "gradient_norm"
    if target_transform in ("logk", "log_k", "logretention", "log_retention"):
        return "logk"
    if target_transform in ("log1p", "log_rt", "logrt"):
        return "log1p"
    return "none"

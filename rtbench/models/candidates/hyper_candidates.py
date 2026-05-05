from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor, early_stopping
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from ...hyper import head_prior, mol_embeddings, ridge_prior_fit_predict
from ...metrics import compute_metrics
from ..calibration import apply_calibration, calibrate_linear, optimize_weights
from ..trees import CandidateOutput, _inverse_target
from .common import CandidateBuildContext


def _target_cp_ref(ctx: CandidateBuildContext, bundle: Any) -> np.ndarray:
    if ctx.target_cp_reference is not None:
        return np.asarray(ctx.target_cp_reference, dtype=np.float32)
    if len(ctx.X_train_cp):
        return np.asarray(ctx.X_train_cp[0], dtype=np.float32)
    return np.zeros_like(np.asarray(bundle.cp_mean, dtype=np.float32))


def _target_embeddings(bundle: Any, X_mol: np.ndarray, cp_ref: np.ndarray, dataset_id: str | None = None) -> np.ndarray:
    return mol_embeddings(
        bundle,
        X_mol,
        cp_ref if bool(getattr(bundle, "use_conditioned_embeddings", False)) else None,
        dataset_id=dataset_id if bool(getattr(bundle, "use_task_adapters", False)) else None,
    )


class _EmbeddingAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        hidden = int(max(8, hidden))
        bottleneck = int(max(4, min(bottleneck, hidden)))
        self.stem = nn.Sequential(
            nn.Linear(int(in_dim), hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.down = nn.Linear(hidden, bottleneck)
        self.up = nn.Linear(bottleneck, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = h + self.up(torch.relu(self.down(h)))
        return self.out(torch.relu(h)).squeeze(-1)


def build_hyper_candidates(ctx: CandidateBuildContext) -> list[CandidateOutput]:
    if ctx.hyper_bundle is None or not bool(ctx.model_cfg.get("ENABLE_HYPER_TL", False)):
        return []

    bundles = ctx.hyper_bundle if isinstance(ctx.hyper_bundle, list) else [ctx.hyper_bundle]
    hyper_cfg = dict(ctx.model_cfg.get("HYPER_TL", {}) or {})
    ensemble_lambdas = bool(hyper_cfg.get("ensemble_lambdas", False))
    candidates: list[CandidateOutput] = []
    ensemble_val: list[np.ndarray] = []
    ensemble_test: list[np.ndarray] = []
    semantic_bases: list[tuple[str, np.ndarray, np.ndarray]] = []

    for bundle in bundles:
        cp_ref = _target_cp_ref(ctx, bundle)
        Z_tr = _target_embeddings(bundle, ctx.X_train_mol, cp_ref, ctx.target_dataset_id)
        Z_va = _target_embeddings(bundle, ctx.X_val_mol, cp_ref, ctx.target_dataset_id)
        Z_te = _target_embeddings(bundle, ctx.X_test_mol, cp_ref, ctx.target_dataset_id)
        w0, b0 = head_prior(bundle, cp_ref)
        Z_tv = np.concatenate([Z_tr, Z_va], axis=0)
        y_tv = np.concatenate([ctx.y_train, ctx.y_val_used], axis=0)
        train_weights = np.full(len(ctx.y_train), float(ctx.target_weight), dtype=np.float32)
        train_val_weights = np.concatenate(
            [train_weights, np.ones(len(ctx.y_val_used), dtype=np.float32)],
            axis=0,
        )

        if ensemble_lambdas and len(bundle.ridge_lambdas) > 1:
            val_preds = [
                _inverse_target(
                    ridge_prior_fit_predict(
                        Z_train=Z_tr,
                        y_train=ctx.y_train,
                        Z_eval=Z_va,
                        w0=w0,
                        b0=b0,
                        lam=float(lam),
                        lam_b=float(bundle.ridge_lambda_b),
                        sample_weight=train_weights,
                    ),
                    ctx.target_transform,
                    ctx.target_inv_scale,
                    ctx.target_t0_sec,
                )
                for lam in bundle.ridge_lambdas
            ]
            val_mat = np.column_stack(val_preds)
            w_lam = optimize_weights(ctx.y_val_sec, val_mat)
            val_pred = (val_mat @ w_lam).astype(np.float32)

            test_preds = [
                _inverse_target(
                    ridge_prior_fit_predict(
                        Z_train=Z_tv,
                        y_train=y_tv,
                        Z_eval=Z_te,
                        w0=w0,
                        b0=b0,
                        lam=float(lam),
                        lam_b=float(bundle.ridge_lambda_b),
                        sample_weight=train_val_weights,
                    ),
                    ctx.target_transform,
                    ctx.target_inv_scale,
                    ctx.target_t0_sec,
                )
                for lam in bundle.ridge_lambdas
            ]
            test_pred = (np.column_stack(test_preds) @ w_lam).astype(np.float32)
        else:
            best_lambda = None
            best_mae = float("inf")
            best_val = None
            for lam in bundle.ridge_lambdas:
                val_pred = _inverse_target(
                    ridge_prior_fit_predict(
                        Z_train=Z_tr,
                        y_train=ctx.y_train,
                        Z_eval=Z_va,
                        w0=w0,
                        b0=b0,
                        lam=float(lam),
                        lam_b=float(bundle.ridge_lambda_b),
                        sample_weight=train_weights,
                    ),
                    ctx.target_transform,
                    ctx.target_inv_scale,
                    ctx.target_t0_sec,
                )
                mae = float(compute_metrics(ctx.y_val_sec, val_pred)["mae"])
                if mae < best_mae:
                    best_mae = mae
                    best_lambda = float(lam)
                    best_val = val_pred
            if best_lambda is None or best_val is None:
                continue
            val_pred = best_val
            test_pred = _inverse_target(
                ridge_prior_fit_predict(
                    Z_train=Z_tv,
                    y_train=y_tv,
                    Z_eval=Z_te,
                    w0=w0,
                    b0=b0,
                    lam=best_lambda,
                    lam_b=float(bundle.ridge_lambda_b),
                    sample_weight=train_val_weights,
                ),
                ctx.target_transform,
                ctx.target_inv_scale,
                ctx.target_t0_sec,
            )

        ensemble_val.append(val_pred)
        ensemble_test.append(test_pred)

    if ensemble_val and ensemble_test:
        val_pred = np.median(np.stack(ensemble_val, axis=0), axis=0).astype(np.float32)
        test_pred = np.median(np.stack(ensemble_test, axis=0), axis=0).astype(np.float32)
        base_hyper_val_pred = val_pred.copy()
        base_hyper_test_pred = test_pred.copy()
        candidates.append(
            CandidateOutput(
                name=f"HYPER_TL_ENS(n={len(ensemble_val)},lam_ens={ensemble_lambdas})",
                val_pred=val_pred,
                test_pred=test_pred,
                val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                model={"n": len(ensemble_val)},
            )
        )
        semantic_bases.append(("HYPER", base_hyper_val_pred, base_hyper_test_pred))
    else:
        base_hyper_val_pred = None
        base_hyper_test_pred = None

    if bool(ctx.model_cfg.get("ENABLE_HYPER_PRIOR_CAL", False)):
        candidates.extend(_build_hyper_prior_calibration_candidates(ctx, bundles))
    if bool(ctx.model_cfg.get("ENABLE_SHEET_PRIOR_CAL", False)):
        candidates.extend(_build_sheet_prior_calibration_candidates(ctx, bundles))

    if bool(ctx.model_cfg.get("ENABLE_HYPER_EMB_LGBM", False)) and bundles:
        candidates.append(_build_hyper_emb_lgbm(ctx, bundles[0]))
    if bool(ctx.model_cfg.get("ENABLE_HYPER_EMB_ADAPTER", False)) and bundles:
        adapter_candidate = _build_hyper_emb_adapter(ctx, bundles[0])
        if adapter_candidate is not None:
            candidates.append(adapter_candidate)
    if bool(ctx.model_cfg.get("ENABLE_SHEET_EMB_LGBM", False)) and bundles:
        sheet_candidate = _build_sheet_emb_lgbm(ctx, bundles[0])
        if sheet_candidate is not None:
            candidates.append(sheet_candidate)
            semantic_bases.append(
                (
                    "SHEET_EMB",
                    np.asarray(sheet_candidate.val_pred, dtype=np.float32),
                    np.asarray(sheet_candidate.test_pred, dtype=np.float32),
                )
            )
    if bool(ctx.model_cfg.get("ENABLE_SHEET_EMB_LGBM_SEC", False)) and bundles:
        sheet_sec_candidate = _build_sheet_emb_lgbm(ctx, bundles[0], use_seconds=True)
        if sheet_sec_candidate is not None:
            candidates.append(sheet_sec_candidate)
    if bool(ctx.model_cfg.get("ENABLE_SHEET_FULL_LGBM", False)) and bundles:
        full_candidate = _build_sheet_full_lgbm(ctx, bundles[0])
        if full_candidate is not None:
            candidates.append(full_candidate)
    if bool(ctx.model_cfg.get("ENABLE_SHEET_FULL_LGBM_SEC", False)) and bundles:
        full_sec_candidate = _build_sheet_full_lgbm(ctx, bundles[0], use_seconds=True)
        if full_sec_candidate is not None:
            candidates.append(full_sec_candidate)
    if bool(ctx.model_cfg.get("ENABLE_SHEET_RESID_LGBM", False)) and bundles:
        resid_candidate = _build_sheet_residual_lgbm(ctx, bundles[0], use_full=False)
        if resid_candidate is not None:
            candidates.append(resid_candidate)
    if bool(ctx.model_cfg.get("ENABLE_SHEET_FULL_RESID_LGBM", False)) and bundles:
        full_resid_candidate = _build_sheet_residual_lgbm(ctx, bundles[0], use_full=True)
        if full_resid_candidate is not None:
            candidates.append(full_resid_candidate)
    if bool(ctx.model_cfg.get("ENABLE_SHEET_MEMORY_ATTENTION", False)) and bundles:
        memory_candidate = _build_sheet_memory_attention(ctx, bundles[0])
        if memory_candidate is not None:
            candidates.append(memory_candidate)
    if bool(ctx.model_cfg.get("ENABLE_RETENTION_ORDER_INDEX", False)):
        roi_candidates = _build_retention_order_index_candidates(ctx)
        candidates.extend(roi_candidates)
    if bool(ctx.model_cfg.get("ENABLE_SEMANTIC_CLASS_PRIOR", False)):
        for base_name, base_val, base_test in semantic_bases:
            semantic_candidates = _build_semantic_class_prior_candidates(
                ctx,
                base_name=base_name,
                base_val_pred=base_val,
                base_test_pred=base_test,
            )
            candidates.extend(semantic_candidates)
    if bool(ctx.model_cfg.get("ENABLE_TARGET_CLASS_QUANTILE_RULES", False)):
        for base_name, base_val, base_test in semantic_bases:
            quantile_candidates = _build_target_class_quantile_rule_candidates(
                ctx,
                base_name=base_name,
                base_val_pred=base_val,
                base_test_pred=base_test,
            )
            candidates.extend(quantile_candidates)
    if bool(ctx.model_cfg.get("ENABLE_CANDIDATE_SIMILARITY_DIAGNOSTICS", False)) and bundles:
        _attach_candidate_similarity_diagnostics(ctx, candidates, bundles[0])
    return candidates


def _attach_candidate_similarity_diagnostics(
    ctx: CandidateBuildContext,
    candidates: list[CandidateOutput],
    bundle: Any,
) -> None:
    diagnostics = _candidate_similarity_diagnostics(ctx, bundle)
    if not diagnostics:
        return
    for candidate in candidates:
        if isinstance(candidate.model, dict):
            candidate.model = {**candidate.model, "similarity_diagnostics": diagnostics}


def _candidate_similarity_diagnostics(ctx: CandidateBuildContext, bundle: Any) -> dict[str, Any]:
    max_ref_rows = int(ctx.model_cfg.get("CANDIDATE_SIMILARITY_DIAGNOSTICS_MAX_REF_ROWS", 2000) or 2000)
    cp_ref = _target_cp_ref(ctx, bundle)
    spaces: dict[str, tuple[np.ndarray, dict[str, np.ndarray | None]]] = {
        "descriptor": (
            np.asarray(ctx.X_test_mol, dtype=np.float32),
            {
                "train": np.asarray(ctx.X_train_mol, dtype=np.float32),
                "val": np.asarray(ctx.X_val_mol, dtype=np.float32),
                "source": np.asarray(ctx.X_src_mol, dtype=np.float32),
            },
        ),
        "cp": (
            np.asarray(
                ctx.X_test_cp if ctx.X_test_cp is not None else np.empty((len(ctx.X_test_mol), 0)),
                dtype=np.float32,
            ),
            {
                "train": np.asarray(ctx.X_train_cp, dtype=np.float32),
                "val": np.asarray(ctx.X_val_cp if ctx.X_val_cp is not None else np.empty((0, 0)), dtype=np.float32),
                "source": np.asarray(ctx.X_src_cp, dtype=np.float32),
            },
        ),
        "embedding": (
            _target_embeddings(bundle, ctx.X_test_mol, cp_ref, ctx.target_dataset_id),
            {
                "train": _target_embeddings(bundle, ctx.X_train_mol, cp_ref, ctx.target_dataset_id),
                "val": _target_embeddings(bundle, ctx.X_val_mol, cp_ref, ctx.target_dataset_id),
                "source": _target_embeddings(bundle, ctx.X_src_mol, cp_ref, None),
            },
        ),
    }
    out: dict[str, Any] = {"n_test": int(len(ctx.X_test_mol)), "spaces": {}}
    for space_name, (test_x, refs) in spaces.items():
        space_diag: dict[str, Any] = {}
        for ref_name, ref_x in refs.items():
            summary = _distance_percentile_summary(test_x, ref_x, max_ref_rows=max_ref_rows)
            if summary:
                space_diag[f"test_vs_{ref_name}"] = summary
        if space_diag:
            out["spaces"][space_name] = space_diag
    return out if out["spaces"] else {}


def _distance_percentile_summary(
    eval_x: np.ndarray,
    ref_x: np.ndarray | None,
    *,
    max_ref_rows: int = 2000,
) -> dict[str, float] | None:
    eval_arr = _as_2d_float(eval_x)
    ref_arr = _as_2d_float(ref_x)
    if eval_arr is None or ref_arr is None or len(eval_arr) == 0 or len(ref_arr) == 0:
        return None
    if eval_arr.shape[1] != ref_arr.shape[1]:
        return None
    ref_arr = _limit_similarity_reference(ref_arr, max_ref_rows)
    eval_scaled, ref_scaled = _standardize_eval_and_ref(eval_arr, ref_arr)
    eval_nn = _nearest_distances(eval_scaled, ref_scaled)
    ref_baseline = _reference_nearest_distances(ref_scaled)
    if len(eval_nn) == 0 or len(ref_baseline) == 0:
        return None
    pct = np.searchsorted(np.sort(ref_baseline), eval_nn, side="right").astype(np.float32) / float(len(ref_baseline))
    return {
        "n_ref": float(len(ref_scaled)),
        "test_nn_distance_median": _finite_stat(eval_nn, "median"),
        "test_nn_distance_p90": _finite_quantile(eval_nn, 0.9),
        "test_nn_distance_max": _finite_stat(eval_nn, "max"),
        "test_distance_percentile_median": _finite_stat(pct, "median"),
        "test_distance_percentile_p90": _finite_quantile(pct, 0.9),
        "test_distance_percentile_max": _finite_stat(pct, "max"),
    }


def _as_2d_float(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        return None
    return arr


def _limit_similarity_reference(ref_x: np.ndarray, max_ref_rows: int) -> np.ndarray:
    if max_ref_rows <= 0 or len(ref_x) <= max_ref_rows:
        return ref_x
    idx = np.linspace(0, len(ref_x) - 1, num=max_ref_rows, dtype=np.int64)
    return ref_x[idx]


def _standardize_eval_and_ref(eval_x: np.ndarray, ref_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(ref_x, axis=0, keepdims=True)
    std = np.std(ref_x, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((eval_x - mean) / std).astype(np.float32), ((ref_x - mean) / std).astype(np.float32)


def _nearest_distances(eval_x: np.ndarray, ref_x: np.ndarray, *, chunk_size: int = 512) -> np.ndarray:
    if len(eval_x) == 0 or len(ref_x) == 0:
        return np.array([], dtype=np.float32)
    out = np.empty(len(eval_x), dtype=np.float32)
    ref_sq = np.sum(np.square(ref_x, dtype=np.float32), axis=1, keepdims=True).T
    for start in range(0, len(eval_x), chunk_size):
        chunk = eval_x[start : start + chunk_size]
        dist_sq = np.sum(np.square(chunk, dtype=np.float32), axis=1, keepdims=True) + ref_sq - 2.0 * (
            chunk @ ref_x.T
        )
        out[start : start + len(chunk)] = np.sqrt(np.maximum(np.min(dist_sq, axis=1), 0.0))
    return out


def _reference_nearest_distances(ref_x: np.ndarray, *, chunk_size: int = 512) -> np.ndarray:
    if len(ref_x) <= 1:
        return np.zeros(len(ref_x), dtype=np.float32)
    out = np.empty(len(ref_x), dtype=np.float32)
    ref_sq = np.sum(np.square(ref_x, dtype=np.float32), axis=1, keepdims=True).T
    for start in range(0, len(ref_x), chunk_size):
        chunk = ref_x[start : start + chunk_size]
        dist_sq = np.sum(np.square(chunk, dtype=np.float32), axis=1, keepdims=True) + ref_sq - 2.0 * (
            chunk @ ref_x.T
        )
        row_idx = np.arange(len(chunk))
        dist_sq[row_idx, start + row_idx] = np.inf
        out[start : start + len(chunk)] = np.sqrt(np.maximum(np.min(dist_sq, axis=1), 0.0))
    return out


def _finite_quantile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.quantile(arr, float(q)))


def _split_target_tokens(ctx: CandidateBuildContext) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]], list[tuple[str, ...]]]:
    total = len(ctx.y_train) + len(ctx.y_val_sec) + len(ctx.X_test)
    tokens = list(ctx.target_context_tokens or [])
    if len(tokens) != total:
        tokens = [tuple()] * total
    n_tr = len(ctx.y_train)
    n_va = len(ctx.y_val_sec)
    return tokens[:n_tr], tokens[n_tr : n_tr + n_va], tokens[n_tr + n_va :]


def _semantic_required_patterns(raw: Any) -> list[re.Pattern[str]]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        values = [raw]
    else:
        try:
            values = list(raw)
        except TypeError:
            values = [raw]
    patterns: list[re.Pattern[str]] = []
    for value in values:
        text = str(value).strip()
        if text:
            patterns.append(re.compile(text, flags=re.IGNORECASE))
    return patterns


def _tokens_match_all_required(tokens: list[tuple[str, ...]], patterns: list[re.Pattern[str]]) -> bool:
    if not patterns:
        return True
    flat = [str(tok) for row in tokens for tok in row]
    if not flat:
        return False
    return all(any(pattern.search(tok) for tok in flat) for pattern in patterns)


def _semantic_target_quantile_rules(raw: Any) -> list[tuple[re.Pattern[str], float]]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        values = [raw]
    else:
        try:
            values = list(raw)
        except TypeError:
            values = [raw]
    rules: list[tuple[re.Pattern[str], float]] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if "=>" in text:
            pattern_text, quantile_text = text.rsplit("=>", 1)
        elif "::" in text:
            pattern_text, quantile_text = text.rsplit("::", 1)
        else:
            continue
        pattern_text = pattern_text.strip()
        quantile_text = quantile_text.strip()
        if not pattern_text or not quantile_text:
            continue
        rules.append((re.compile(pattern_text, flags=re.IGNORECASE), float(quantile_text)))
    return rules


def _cp_cosine_weights(source_cp: np.ndarray, target_cp: np.ndarray, power: float) -> np.ndarray:
    X = np.asarray(source_cp, dtype=np.float32)
    ref = np.asarray(target_cp, dtype=np.float32).reshape(-1)
    if X.ndim != 2 or X.shape[0] == 0 or ref.size != X.shape[1]:
        return np.ones(X.shape[0] if X.ndim == 2 else 0, dtype=np.float32)
    Xn = np.linalg.norm(X, axis=1)
    rn = float(np.linalg.norm(ref))
    if rn <= 0.0:
        return np.ones(X.shape[0], dtype=np.float32)
    sim = (X @ ref) / np.maximum(Xn * rn, 1e-8)
    sim = np.clip((sim + 1.0) * 0.5, 0.0, 1.0)
    return np.power(sim, float(max(power, 0.0))).astype(np.float32)


def _semantic_prior_raw(
    *,
    target_tokens: list[tuple[str, ...]],
    target_keys: np.ndarray | None,
    source_tokens: list[tuple[str, ...]],
    source_keys: np.ndarray | None,
    source_y_used: np.ndarray,
    source_weights: np.ndarray,
    min_support: int,
    exclude_exact_key: bool,
    agg: str = "mean",
    quantile: float = 0.8,
    token_pattern: re.Pattern[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y_src = np.asarray(source_y_used, dtype=np.float32).reshape(-1)
    priors = np.full(len(target_tokens), np.nan, dtype=np.float32)
    support = np.zeros(len(target_tokens), dtype=np.float32)
    if len(source_tokens) != len(y_src):
        return priors, support
    token_to_idx: dict[str, list[int]] = {}
    for i, toks in enumerate(source_tokens):
        for tok in toks:
            token_to_idx.setdefault(str(tok), []).append(i)
    source_keys_arr = np.asarray(source_keys, dtype=object) if source_keys is not None else None
    target_keys_arr = np.asarray(target_keys, dtype=object) if target_keys is not None else None
    for row_idx, toks in enumerate(target_tokens):
        row_tokens = tuple(toks)
        if token_pattern is not None:
            row_tokens = tuple(tok for tok in toks if token_pattern.search(str(tok)))
            if not row_tokens:
                continue
        exact_key = ""
        if target_keys_arr is not None and row_idx < len(target_keys_arr):
            exact_key = str(target_keys_arr[row_idx]).strip()
        for tok in row_tokens:
            idxs = token_to_idx.get(str(tok), [])
            if not idxs:
                continue
            if exclude_exact_key and exact_key and source_keys_arr is not None:
                idxs = [idx for idx in idxs if str(source_keys_arr[idx]).strip() != exact_key]
            if len(idxs) < int(min_support):
                continue
            w = np.asarray(source_weights[idxs], dtype=np.float64)
            if not np.any(np.isfinite(w)) or float(np.nansum(w)) <= 1e-12:
                w = np.ones(len(idxs), dtype=np.float64)
            vals = y_src[idxs].astype(np.float64)
            finite = np.isfinite(vals)
            if int(np.sum(finite)) < int(min_support):
                continue
            w = w[finite]
            vals = vals[finite]
            if str(agg).strip().lower() in {"quantile", "q", "upper"}:
                priors[row_idx] = float(np.nanquantile(vals, float(np.clip(quantile, 0.0, 1.0))))
            else:
                priors[row_idx] = float(np.sum(vals * w) / max(float(np.sum(w)), 1e-12))
            support[row_idx] = float(len(vals))
            break
    return priors, support


def _calibrate_semantic_prior(prior_sec: np.ndarray, y_sec: np.ndarray, *, min_points: int) -> Any:
    x = np.asarray(prior_sec, dtype=np.float64).reshape(-1)
    y = np.asarray(y_sec, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < int(min_points):
        return None
    A = np.column_stack([x[finite], np.ones(int(np.sum(finite)))])
    coef, *_ = np.linalg.lstsq(A, y[finite], rcond=None)
    slope = float(np.clip(coef[0], 0.25, 2.5))
    intercept = float(coef[1])
    return slope, intercept


def _apply_semantic_calibration(prior_sec: np.ndarray, calib: Any) -> np.ndarray:
    out = np.asarray(prior_sec, dtype=np.float32).copy()
    if calib is None:
        return out
    slope, intercept = calib
    finite = np.isfinite(out)
    out[finite] = (float(slope) * out[finite] + float(intercept)).astype(np.float32)
    return out


def _apply_semantic_target_quantile_floor(prior_sec: np.ndarray, y_ref_sec: np.ndarray, quantile: float) -> np.ndarray:
    out = np.asarray(prior_sec, dtype=np.float32).copy()
    q = float(quantile)
    if q < 0.0:
        return out
    ref = np.asarray(y_ref_sec, dtype=np.float32).reshape(-1)
    finite_ref = ref[np.isfinite(ref)]
    if finite_ref.size == 0:
        return out
    floor_value = float(np.nanquantile(finite_ref, float(np.clip(q, 0.0, 1.0))))
    mask = np.isfinite(out)
    out[mask] = np.maximum(out[mask], floor_value).astype(np.float32)
    return out


def _apply_semantic_target_quantile_rules(
    prior_sec: np.ndarray,
    tokens: list[tuple[str, ...]],
    y_ref_sec: np.ndarray,
    rules: list[tuple[re.Pattern[str], float]],
) -> np.ndarray:
    out = np.asarray(prior_sec, dtype=np.float32).copy()
    if not rules:
        return out
    ref = np.asarray(y_ref_sec, dtype=np.float32).reshape(-1)
    finite_ref = ref[np.isfinite(ref)]
    if finite_ref.size == 0:
        return out
    quantile_cache: dict[float, float] = {}
    for row_idx, row_tokens in enumerate(tokens):
        text = " ".join(str(tok) for tok in row_tokens)
        if not text:
            continue
        for pattern, quantile in rules:
            if not pattern.search(text):
                continue
            q = float(np.clip(float(quantile), 0.0, 1.0))
            if q not in quantile_cache:
                quantile_cache[q] = float(np.nanquantile(finite_ref, q))
            out[row_idx] = quantile_cache[q]
            break
    return out


def _blend_semantic_prior(base: np.ndarray, prior: np.ndarray, blend_weight: float) -> np.ndarray:
    out = np.asarray(base, dtype=np.float32).copy()
    prior = np.asarray(prior, dtype=np.float32).reshape(-1)
    mask = np.isfinite(prior)
    w = float(np.clip(blend_weight, 0.0, 1.0))
    out[mask] = ((1.0 - w) * out[mask] + w * prior[mask]).astype(np.float32)
    return out


def _build_semantic_class_prior_candidates(
    ctx: CandidateBuildContext,
    *,
    base_name: str,
    base_val_pred: np.ndarray | None,
    base_test_pred: np.ndarray | None,
) -> list[CandidateOutput]:
    if base_val_pred is None or base_test_pred is None:
        return []
    source_tokens = list(ctx.source_context_tokens or [])
    if len(source_tokens) != len(ctx.y_src):
        return []
    train_tokens, val_tokens, test_tokens = _split_target_tokens(ctx)
    if not any(train_tokens + val_tokens + test_tokens):
        return []
    target_required_patterns = _semantic_required_patterns(
        ctx.model_cfg.get("SEMANTIC_CLASS_PRIOR_REQUIRE_TARGET_TOKEN_REGEXES")
    )
    if not _tokens_match_all_required(train_tokens + val_tokens + test_tokens, target_required_patterns):
        return []
    required_patterns = _semantic_required_patterns(ctx.model_cfg.get("SEMANTIC_CLASS_PRIOR_REQUIRE_TEST_TOKEN_REGEXES"))
    if not _tokens_match_all_required(test_tokens, required_patterns):
        return []
    target_keys = np.asarray(ctx.target_mol_keys, dtype=object) if ctx.target_mol_keys is not None else None
    n_tr = len(ctx.y_train)
    n_va = len(ctx.y_val_sec)
    train_keys = target_keys[:n_tr] if target_keys is not None else None
    val_keys = target_keys[n_tr : n_tr + n_va] if target_keys is not None else None
    test_keys = target_keys[n_tr + n_va :] if target_keys is not None else None

    cp_ref = ctx.target_cp_reference
    if cp_ref is None:
        cp_ref = ctx.X_train_cp[0] if len(ctx.X_train_cp) else np.zeros(ctx.X_src_cp.shape[1], dtype=np.float32)
    cp_w = _cp_cosine_weights(ctx.X_src_cp, cp_ref, float(ctx.model_cfg.get("SEMANTIC_CLASS_PRIOR_CP_POWER", 4.0)))
    min_support = int(ctx.model_cfg.get("SEMANTIC_CLASS_PRIOR_MIN_SUPPORT", 1))
    exclude_exact = bool(ctx.model_cfg.get("SEMANTIC_CLASS_PRIOR_EXCLUDE_EXACT_KEY", True))
    agg = str(ctx.model_cfg.get("SEMANTIC_CLASS_PRIOR_AGG", "mean"))
    quantile = float(ctx.model_cfg.get("SEMANTIC_CLASS_PRIOR_QUANTILE", 0.8))
    token_regex = str(ctx.model_cfg.get("SEMANTIC_CLASS_PRIOR_TOKEN_REGEX", "") or "").strip()
    token_pattern = re.compile(token_regex, flags=re.IGNORECASE) if token_regex else None
    use_source_seconds = bool(ctx.model_cfg.get("SEMANTIC_CLASS_PRIOR_SOURCE_SECONDS", False))
    source_y_for_prior = ctx.y_src_sec if use_source_seconds else ctx.y_src

    prior_train_used, _ = _semantic_prior_raw(
        target_tokens=train_tokens,
        target_keys=train_keys,
        source_tokens=source_tokens,
        source_keys=ctx.source_mol_keys,
        source_y_used=source_y_for_prior,
        source_weights=cp_w,
        min_support=min_support,
        exclude_exact_key=exclude_exact,
        agg=agg,
        quantile=quantile,
        token_pattern=token_pattern,
    )
    prior_val_used, val_support = _semantic_prior_raw(
        target_tokens=val_tokens,
        target_keys=val_keys,
        source_tokens=source_tokens,
        source_keys=ctx.source_mol_keys,
        source_y_used=source_y_for_prior,
        source_weights=cp_w,
        min_support=min_support,
        exclude_exact_key=exclude_exact,
        agg=agg,
        quantile=quantile,
        token_pattern=token_pattern,
    )
    prior_test_used, test_support = _semantic_prior_raw(
        target_tokens=test_tokens,
        target_keys=test_keys,
        source_tokens=source_tokens,
        source_keys=ctx.source_mol_keys,
        source_y_used=source_y_for_prior,
        source_weights=cp_w,
        min_support=min_support,
        exclude_exact_key=exclude_exact,
        agg=agg,
        quantile=quantile,
        token_pattern=token_pattern,
    )
    if use_source_seconds:
        prior_train_sec = np.asarray(prior_train_used, dtype=np.float32).copy()
        prior_val_sec = np.asarray(prior_val_used, dtype=np.float32).copy()
        prior_test_sec = np.asarray(prior_test_used, dtype=np.float32).copy()
    else:
        prior_train_sec = _inverse_target(prior_train_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
        prior_val_sec = _inverse_target(prior_val_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
        prior_test_sec = _inverse_target(prior_test_used, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    prior_val_sec[~np.isfinite(prior_val_used)] = np.nan
    prior_test_sec[~np.isfinite(prior_test_used)] = np.nan
    prior_train_sec[~np.isfinite(prior_train_used)] = np.nan

    min_cal = int(ctx.model_cfg.get("SEMANTIC_CLASS_PRIOR_MIN_CAL_POINTS", 3))
    calib_val = _calibrate_semantic_prior(prior_train_sec, ctx.y_train_sec, min_points=min_cal)
    prior_val_cal = _apply_semantic_calibration(prior_val_sec, calib_val)
    y_tv_sec = np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0)
    prior_tv_sec = np.concatenate([prior_train_sec, prior_val_sec], axis=0)
    calib_test = _calibrate_semantic_prior(prior_tv_sec, y_tv_sec, min_points=min_cal)
    prior_test_cal = _apply_semantic_calibration(prior_test_sec, calib_test)
    quantile_floor = float(ctx.model_cfg.get("SEMANTIC_CLASS_PRIOR_TARGET_QUANTILE_FLOOR", -1.0))
    prior_val_cal = _apply_semantic_target_quantile_floor(prior_val_cal, ctx.y_train_sec, quantile_floor)
    prior_test_cal = _apply_semantic_target_quantile_floor(prior_test_cal, y_tv_sec, quantile_floor)
    quantile_rules = _semantic_target_quantile_rules(ctx.model_cfg.get("SEMANTIC_CLASS_TARGET_QUANTILE_RULES"))
    prior_val_cal = _apply_semantic_target_quantile_rules(prior_val_cal, val_tokens, ctx.y_train_sec, quantile_rules)
    prior_test_cal = _apply_semantic_target_quantile_rules(prior_test_cal, test_tokens, y_tv_sec, quantile_rules)
    if not (np.any(np.isfinite(prior_val_cal)) or np.any(np.isfinite(prior_test_cal))):
        return []

    blend = float(ctx.model_cfg.get("SEMANTIC_CLASS_PRIOR_BLEND", 0.85))
    val_pred = _blend_semantic_prior(base_val_pred, prior_val_cal, blend)
    test_pred = _blend_semantic_prior(base_test_pred, prior_test_cal, blend)
    model = {
        "val_prior_rows": int(np.sum(np.isfinite(prior_val_cal))),
        "test_prior_rows": int(np.sum(np.isfinite(prior_test_cal))),
        "mean_val_support": float(np.nanmean(val_support)) if np.any(val_support > 0) else 0.0,
        "mean_test_support": float(np.nanmean(test_support)) if np.any(test_support > 0) else 0.0,
        "source_seconds": use_source_seconds,
        "target_quantile_floor": quantile_floor,
        "target_quantile_rules": len(quantile_rules),
    }
    return [
        CandidateOutput(
            name=f"SEMANTIC_CLASS_PRIOR_BLEND_{base_name}",
            val_pred=val_pred.astype(np.float32),
            test_pred=test_pred.astype(np.float32),
            val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
            model=model,
        )
    ]


def _build_target_class_quantile_rule_candidates(
    ctx: CandidateBuildContext,
    *,
    base_name: str,
    base_val_pred: np.ndarray | None,
    base_test_pred: np.ndarray | None,
) -> list[CandidateOutput]:
    if base_val_pred is None or base_test_pred is None:
        return []
    train_tokens, val_tokens, test_tokens = _split_target_tokens(ctx)
    rules = _semantic_target_quantile_rules(ctx.model_cfg.get("TARGET_CLASS_QUANTILE_RULES"))
    if not rules:
        return []
    target_required_patterns = _semantic_required_patterns(
        ctx.model_cfg.get("TARGET_CLASS_QUANTILE_REQUIRE_TARGET_TOKEN_REGEXES")
    )
    if not _tokens_match_all_required(train_tokens + val_tokens + test_tokens, target_required_patterns):
        return []
    required_patterns = _semantic_required_patterns(ctx.model_cfg.get("TARGET_CLASS_QUANTILE_REQUIRE_TEST_TOKEN_REGEXES"))
    if not _tokens_match_all_required(test_tokens, required_patterns):
        return []
    val_prior = np.full(len(ctx.y_val_sec), np.nan, dtype=np.float32)
    test_prior = np.full(len(ctx.X_test), np.nan, dtype=np.float32)
    y_tv_sec = np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0)
    val_prior = _apply_semantic_target_quantile_rules(val_prior, val_tokens, ctx.y_train_sec, rules)
    test_prior = _apply_semantic_target_quantile_rules(test_prior, test_tokens, y_tv_sec, rules)
    if not (np.any(np.isfinite(val_prior)) or np.any(np.isfinite(test_prior))):
        return []
    blend = float(ctx.model_cfg.get("TARGET_CLASS_QUANTILE_BLEND", 1.0))
    val_pred = _blend_semantic_prior(base_val_pred, val_prior, blend)
    test_pred = _blend_semantic_prior(base_test_pred, test_prior, blend)
    model = {
        "val_rule_rows": int(np.sum(np.isfinite(val_prior))),
        "test_rule_rows": int(np.sum(np.isfinite(test_prior))),
        "target_quantile_rules": len(rules),
    }
    return [
        CandidateOutput(
            name=f"TARGET_CLASS_QUANTILE_RULES_{base_name}",
            val_pred=val_pred.astype(np.float32),
            test_pred=test_pred.astype(np.float32),
            val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
            model=model,
        )
    ]


def _rank01(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(len(y))
    if n == 0:
        return np.empty(0, dtype=np.float32)
    if n == 1:
        return np.full(1, 0.5, dtype=np.float32)
    ranks = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64) - 1.0
    return np.clip(ranks / float(max(n - 1, 1)), 0.0, 1.0).astype(np.float32)


def _rank01_by_dataset(values: np.ndarray, dataset_ids: np.ndarray | None) -> np.ndarray:
    y = np.asarray(values, dtype=np.float32).reshape(-1)
    if dataset_ids is None or len(dataset_ids) != len(y):
        return _rank01(y)
    ids = np.asarray(dataset_ids, dtype=object)
    out = np.zeros(len(y), dtype=np.float32)
    for ds in sorted({str(value) for value in ids.tolist()}):
        mask = np.asarray([str(value) == ds for value in ids.tolist()], dtype=bool)
        out[mask] = _rank01(y[mask])
    return out


def _roi_feature_matrix(ctx: CandidateBuildContext, split: str) -> np.ndarray:
    feature_mode = str(ctx.model_cfg.get("RETENTION_ORDER_FEATURES", "full")).strip().lower()
    if split == "source":
        if feature_mode == "mol":
            return np.asarray(ctx.X_src_mol, dtype=np.float32)
        if feature_mode == "mol_cp":
            return np.concatenate([ctx.X_src_mol, ctx.X_src_cp], axis=1).astype(np.float32)
        return np.asarray(ctx.X_src, dtype=np.float32)
    if split == "train":
        if feature_mode == "mol":
            return np.asarray(ctx.X_train_mol, dtype=np.float32)
        if feature_mode == "mol_cp":
            return np.concatenate([ctx.X_train_mol, ctx.X_train_cp], axis=1).astype(np.float32)
        return np.asarray(ctx.X_train, dtype=np.float32)
    if split == "val":
        if feature_mode == "mol":
            return np.asarray(ctx.X_val_mol, dtype=np.float32)
        if feature_mode == "mol_cp":
            if ctx.X_val_cp is None:
                raise ValueError("X_val_cp is required for RETENTION_ORDER_FEATURES=mol_cp")
            return np.concatenate([ctx.X_val_mol, ctx.X_val_cp], axis=1).astype(np.float32)
        return np.asarray(ctx.X_val, dtype=np.float32)
    if feature_mode == "mol":
        return np.asarray(ctx.X_test_mol, dtype=np.float32)
    if feature_mode == "mol_cp":
        if ctx.X_test_cp is None:
            raise ValueError("X_test_cp is required for RETENTION_ORDER_FEATURES=mol_cp")
        return np.concatenate([ctx.X_test_mol, ctx.X_test_cp], axis=1).astype(np.float32)
    return np.asarray(ctx.X_test, dtype=np.float32)


def _fit_roi_mapper(pred_roi: np.ndarray, y_sec: np.ndarray, *, mode: str) -> Any:
    x = np.asarray(pred_roi, dtype=np.float64).reshape(-1)
    y = np.asarray(y_sec, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 2:
        center = float(np.nanmedian(y)) if len(y) else 0.0
        return ("constant", center)
    x_fit = x[finite]
    y_fit = y[finite]
    if str(mode).strip().lower() == "linear":
        A = np.column_stack([x_fit, np.ones_like(x_fit)])
        coef, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
        return ("linear", float(coef[0]), float(coef[1]))
    order = np.argsort(x_fit)
    mapper = IsotonicRegression(out_of_bounds="clip", increasing=True)
    mapper.fit(x_fit[order], y_fit[order])
    return mapper


def _apply_roi_mapper(mapper: Any, pred_roi: np.ndarray) -> np.ndarray:
    x = np.asarray(pred_roi, dtype=np.float64).reshape(-1)
    if isinstance(mapper, tuple):
        if mapper[0] == "constant":
            return np.full(len(x), float(mapper[1]), dtype=np.float32)
        if mapper[0] == "linear":
            return (float(mapper[1]) * x + float(mapper[2])).astype(np.float32)
    return np.asarray(mapper.predict(x), dtype=np.float32)


def _build_retention_order_index_candidates(ctx: CandidateBuildContext) -> list[CandidateOutput]:
    if len(ctx.y_train_sec) < 3 or len(ctx.X_src) < 10:
        return []

    params = dict(ctx.model_cfg.get("RETENTION_ORDER_LGBM", {}) or {})
    if not params:
        params = {
            "n_estimators": int(ctx.model_cfg.get("RETENTION_ORDER_N_ESTIMATORS", 900)),
            "learning_rate": float(ctx.model_cfg.get("RETENTION_ORDER_LEARNING_RATE", 0.03)),
            "num_leaves": int(ctx.model_cfg.get("RETENTION_ORDER_NUM_LEAVES", 31)),
            "objective": str(ctx.model_cfg.get("RETENTION_ORDER_OBJECTIVE", "mae")),
            "feature_fraction": float(ctx.model_cfg.get("RETENTION_ORDER_FEATURE_FRACTION", 0.85)),
            "bagging_fraction": float(ctx.model_cfg.get("RETENTION_ORDER_BAGGING_FRACTION", 0.9)),
            "bagging_freq": int(ctx.model_cfg.get("RETENTION_ORDER_BAGGING_FREQ", 1)),
            "min_child_samples": int(ctx.model_cfg.get("RETENTION_ORDER_MIN_CHILD_SAMPLES", 8)),
        }
    target_in_model = bool(ctx.model_cfg.get("RETENTION_ORDER_USE_TARGET_TRAIN", True))
    source_weight = float(ctx.model_cfg.get("RETENTION_ORDER_SOURCE_WEIGHT", ctx.source_weight))
    target_weight = float(ctx.model_cfg.get("RETENTION_ORDER_TARGET_WEIGHT", ctx.target_weight))
    stop_rounds = int(ctx.model_cfg.get("RETENTION_ORDER_EARLY_STOPPING_ROUNDS", 0) or 0)
    mapper_modes_raw = ctx.model_cfg.get("RETENTION_ORDER_MAPPERS", ["isotonic"])
    if isinstance(mapper_modes_raw, (str, bytes)):
        mapper_modes = [str(mapper_modes_raw)]
    else:
        mapper_modes = [str(value) for value in mapper_modes_raw]
    mapper_modes = [mode for mode in mapper_modes if mode.strip()]
    if not mapper_modes:
        mapper_modes = ["isotonic"]

    X_src = _roi_feature_matrix(ctx, "source")
    X_tr = _roi_feature_matrix(ctx, "train")
    X_va = _roi_feature_matrix(ctx, "val")
    X_te = _roi_feature_matrix(ctx, "test")
    y_src_roi = _rank01_by_dataset(ctx.y_src_sec, ctx.source_row_dataset_ids)
    y_tr_roi = _rank01(ctx.y_train_sec)
    y_tv_sec = np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0)
    y_tv_roi = _rank01(y_tv_sec)

    train_parts = [X_src]
    y_parts = [y_src_roi]
    if ctx.source_sample_weights is not None:
        src_w = np.asarray(ctx.source_sample_weights, dtype=np.float32)
    else:
        src_w = np.full(len(X_src), source_weight, dtype=np.float32)
    w_parts = [src_w]
    if target_in_model:
        train_parts.append(X_tr)
        y_parts.append(y_tr_roi)
        w_parts.append(np.full(len(X_tr), target_weight, dtype=np.float32))
    X_fit = np.concatenate(train_parts, axis=0)
    y_fit = np.concatenate(y_parts, axis=0)
    w_fit = np.concatenate(w_parts, axis=0)

    cols = [f"f{i}" for i in range(X_fit.shape[1])]
    model = LGBMRegressor(random_state=int(ctx.seed) + 503, n_jobs=8, verbose=-1, **params)
    callbacks = [early_stopping(stopping_rounds=stop_rounds, verbose=False)] if stop_rounds > 0 and len(X_va) else None
    if callbacks:
        model.fit(
            pd.DataFrame(X_fit, columns=cols),
            y_fit,
            sample_weight=w_fit,
            eval_set=[(pd.DataFrame(X_va, columns=cols), _rank01(ctx.y_val_sec))],
            eval_metric="l1",
            callbacks=callbacks,
        )
        num_it = int(getattr(model, "best_iteration_", 0)) or None
        pred_tr_roi = model.predict(pd.DataFrame(X_tr, columns=cols), num_iteration=num_it)
        pred_va_roi = model.predict(pd.DataFrame(X_va, columns=cols), num_iteration=num_it)
    else:
        model.fit(pd.DataFrame(X_fit, columns=cols), y_fit, sample_weight=w_fit)
        pred_tr_roi = model.predict(pd.DataFrame(X_tr, columns=cols))
        pred_va_roi = model.predict(pd.DataFrame(X_va, columns=cols))

    X_tv = np.concatenate([X_tr, X_va], axis=0)
    refit_parts = [X_src]
    refit_y_parts = [y_src_roi]
    refit_w_parts = [src_w]
    if target_in_model:
        refit_parts.append(X_tv)
        refit_y_parts.append(y_tv_roi)
        refit_w_parts.append(np.full(len(X_tv), target_weight, dtype=np.float32))
    X_refit = np.concatenate(refit_parts, axis=0)
    y_refit = np.concatenate(refit_y_parts, axis=0)
    w_refit = np.concatenate(refit_w_parts, axis=0)
    refit_model = LGBMRegressor(random_state=int(ctx.seed) + 509, n_jobs=8, verbose=-1, **params)
    refit_model.fit(pd.DataFrame(X_refit, columns=cols), y_refit, sample_weight=w_refit)
    pred_tv_roi = refit_model.predict(pd.DataFrame(X_tv, columns=cols))
    pred_te_roi = refit_model.predict(pd.DataFrame(X_te, columns=cols))

    candidates: list[CandidateOutput] = []
    for mode in mapper_modes:
        mapper_val = _fit_roi_mapper(pred_tr_roi, ctx.y_train_sec, mode=mode)
        val_pred = _apply_roi_mapper(mapper_val, pred_va_roi)
        mapper_test = _fit_roi_mapper(pred_tv_roi, y_tv_sec, mode=mode)
        test_pred = _apply_roi_mapper(mapper_test, pred_te_roi)
        candidates.append(
            CandidateOutput(
                name=f"RETENTION_ORDER_INDEX_LGBM_{mode.strip().upper()}",
                val_pred=val_pred,
                test_pred=test_pred,
                val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                model=refit_model,
            )
        )
    return candidates


def _build_hyper_prior_calibration_candidates(ctx: CandidateBuildContext, bundles: list[Any]) -> list[CandidateOutput]:
    """Calibrate source-task hypernetwork priors with target train labels only.

    This keeps validation labels reserved for candidate selection while allowing a
    post-projection model-to-target correction before the normal target ridge
    adaptation is considered.
    """
    if len(ctx.y_train_sec) < 2 or not bundles:
        return []

    train_priors: list[np.ndarray] = []
    val_priors: list[np.ndarray] = []
    test_priors: list[np.ndarray] = []
    for bundle in bundles:
        cp_ref = _target_cp_ref(ctx, bundle)
        try:
            w0, b0 = head_prior(bundle, cp_ref)
            tr = _inverse_target(
                _target_embeddings(bundle, ctx.X_train_mol, cp_ref, ctx.target_dataset_id) @ w0 + b0,
                ctx.target_transform,
                ctx.target_inv_scale,
                ctx.target_t0_sec,
            )
            va = _inverse_target(
                _target_embeddings(bundle, ctx.X_val_mol, cp_ref, ctx.target_dataset_id) @ w0 + b0,
                ctx.target_transform,
                ctx.target_inv_scale,
                ctx.target_t0_sec,
            )
            te = _inverse_target(
                _target_embeddings(bundle, ctx.X_test_mol, cp_ref, ctx.target_dataset_id) @ w0 + b0,
                ctx.target_transform,
                ctx.target_inv_scale,
                ctx.target_t0_sec,
            )
        except Exception:
            continue
        if np.all(np.isfinite(tr)) and np.all(np.isfinite(va)) and np.all(np.isfinite(te)):
            train_priors.append(np.asarray(tr, dtype=np.float32))
            val_priors.append(np.asarray(va, dtype=np.float32))
            test_priors.append(np.asarray(te, dtype=np.float32))

    if not train_priors:
        return []

    modes = [str(x).strip().lower() for x in ctx.model_cfg.get("HYPER_PRIOR_CAL_MODES", ["linear", "isotonic"])]
    modes = [mode for mode in modes if mode]
    candidates: list[CandidateOutput] = []
    for mode in modes:
        val_preds: list[np.ndarray] = []
        test_preds: list[np.ndarray] = []
        val_diags: list[dict[str, float | str]] = []
        test_diags: list[dict[str, float | str]] = []
        for prior_tr, prior_va, prior_te in zip(train_priors, val_priors, test_priors):
            val_fit = _calibrate_prior_with_diagnostics(
                mode=mode,
                fit_pred=prior_tr,
                fit_y=ctx.y_train_sec,
                eval_pred=prior_va,
            )
            if val_fit is None:
                continue
            val_pred, val_diag = val_fit
            test_fit = _calibrate_prior_with_diagnostics(
                mode=mode,
                fit_pred=np.concatenate([prior_tr, prior_va], axis=0),
                fit_y=np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0),
                eval_pred=prior_te,
            )
            if test_fit is None:
                continue
            test_pred, test_diag = test_fit
            val_preds.append(val_pred)
            test_preds.append(test_pred)
            val_diags.append(val_diag)
            test_diags.append(test_diag)
        if not val_preds:
            continue
        val_out = np.median(np.stack(val_preds, axis=0), axis=0).astype(np.float32)
        test_out = np.median(np.stack(test_preds, axis=0), axis=0).astype(np.float32)
        prior_calibration = _summarize_prior_calibration_diagnostics(
            mode=mode,
            val_diags=val_diags,
            test_diags=test_diags,
        )
        candidates.append(
            CandidateOutput(
                name=f"HYPER_PRIOR_CAL_{mode.upper()}(n={len(val_preds)})",
                val_pred=val_out,
                test_pred=test_out,
                val_metrics=compute_metrics(ctx.y_val_sec, val_out),
                model={
                    "type": "hyper_prior_calibration",
                    "mode": mode,
                    "n": len(val_preds),
                    "prior_calibration": prior_calibration,
                },
            )
        )
    return candidates


def _finite_stat(values: np.ndarray, reducer: str) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    if reducer == "min":
        return float(np.min(finite))
    if reducer == "max":
        return float(np.max(finite))
    if reducer == "median":
        return float(np.median(finite))
    return float(np.mean(finite))


def _prior_calibration_base_diagnostics(
    *,
    mode: str,
    fit_pred: np.ndarray,
    fit_y: np.ndarray,
    eval_pred: np.ndarray,
) -> dict[str, float | str]:
    x = np.asarray(fit_pred, dtype=np.float64).reshape(-1)
    y = np.asarray(fit_y, dtype=np.float64).reshape(-1)
    z = np.asarray(eval_pred, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x_fit = x[keep]
    y_fit = y[keep]
    fit_min = _finite_stat(x_fit, "min")
    fit_max = _finite_stat(x_fit, "max")
    eval_min = _finite_stat(z, "min")
    eval_max = _finite_stat(z, "max")
    fit_range = fit_max - fit_min if np.isfinite(fit_min) and np.isfinite(fit_max) else float("nan")
    eval_range = eval_max - eval_min if np.isfinite(eval_min) and np.isfinite(eval_max) else float("nan")
    lower = np.isfinite(z) & np.isfinite(fit_min) & (z < fit_min)
    upper = np.isfinite(z) & np.isfinite(fit_max) & (z > fit_max)
    extrap = lower | upper
    denom = float(max(int(np.sum(np.isfinite(z))), 1))
    lower_amount = float(max(0.0, fit_min - eval_min)) if np.isfinite(fit_min) and np.isfinite(eval_min) else float("nan")
    upper_amount = float(max(0.0, eval_max - fit_max)) if np.isfinite(fit_max) and np.isfinite(eval_max) else float("nan")
    max_extrap = max(
        lower_amount if np.isfinite(lower_amount) else 0.0,
        upper_amount if np.isfinite(upper_amount) else 0.0,
    )
    safe_fit_range = max(float(fit_range), 1e-8) if np.isfinite(fit_range) else float("nan")
    return {
        "mode": str(mode).strip().lower(),
        "fit_n": float(int(np.sum(keep))),
        "fit_pred_min": fit_min,
        "fit_pred_max": fit_max,
        "fit_pred_range": float(fit_range),
        "fit_y_min": _finite_stat(y_fit, "min"),
        "fit_y_max": _finite_stat(y_fit, "max"),
        "fit_y_range": float(_finite_stat(y_fit, "max") - _finite_stat(y_fit, "min")),
        "eval_pred_min": eval_min,
        "eval_pred_max": eval_max,
        "eval_pred_range": float(eval_range),
        "eval_to_fit_range_ratio": float(eval_range / safe_fit_range) if np.isfinite(eval_range) and np.isfinite(safe_fit_range) else float("nan"),
        "eval_extrapolation_lower_frac": float(np.sum(lower) / denom),
        "eval_extrapolation_upper_frac": float(np.sum(upper) / denom),
        "eval_extrapolation_frac": float(np.sum(extrap) / denom),
        "eval_extrapolation_amount": float(max_extrap),
        "eval_extrapolation_ratio": float(max_extrap / safe_fit_range) if np.isfinite(safe_fit_range) else float("nan"),
    }


def _aggregate_prior_calibration_diagnostics(diags: list[dict[str, float | str]]) -> dict[str, float]:
    if not diags:
        return {}
    keys = sorted({key for diag in diags for key, value in diag.items() if isinstance(value, (int, float))})
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(diag[key]) for diag in diags if key in diag], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        out[f"{key}_min"] = float(np.min(finite))
        out[f"{key}_max"] = float(np.max(finite))
        out[f"{key}_mean"] = float(np.mean(finite))
        out[f"{key}_median"] = float(np.median(finite))
    slopes = np.asarray([float(diag["slope"]) for diag in diags if "slope" in diag], dtype=np.float64)
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size:
        out["slope_abs_max"] = float(np.max(np.abs(slopes)))
        out["slope_abs_median"] = float(np.median(np.abs(slopes)))
    return out


def _summarize_prior_calibration_diagnostics(
    *,
    mode: str,
    val_diags: list[dict[str, float | str]],
    test_diags: list[dict[str, float | str]],
) -> dict[str, Any]:
    return {
        "mode": str(mode).strip().lower(),
        "n": int(len(test_diags)),
        "val": _aggregate_prior_calibration_diagnostics(val_diags),
        "test": _aggregate_prior_calibration_diagnostics(test_diags),
        "bundles": {
            "val": val_diags,
            "test": test_diags,
        },
    }


def _calibrate_prior_with_diagnostics(
    *,
    mode: str,
    fit_pred: np.ndarray,
    fit_y: np.ndarray,
    eval_pred: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | str]] | None:
    x = np.asarray(fit_pred, dtype=np.float64).reshape(-1)
    y = np.asarray(fit_y, dtype=np.float64).reshape(-1)
    z = np.asarray(eval_pred, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(keep)) < 2 or not np.all(np.isfinite(z)):
        return None
    x = x[keep]
    y = y[keep]
    mode = str(mode).strip().lower()
    diagnostics = _prior_calibration_base_diagnostics(
        mode=mode,
        fit_pred=x,
        fit_y=y,
        eval_pred=z,
    )
    if mode in ("linear", "affine"):
        a, b = calibrate_linear(y, x)
        diagnostics.update({"slope": float(a), "intercept": float(b)})
        return apply_calibration(a, b, z), diagnostics
    if mode in ("isotonic", "iso"):
        if len(np.unique(x)) < 2:
            return None
        model = IsotonicRegression(out_of_bounds="clip")
        try:
            model.fit(x, y)
            pred = np.asarray(model.predict(z), dtype=np.float32)
            thresholds_x = np.asarray(getattr(model, "X_thresholds_", []), dtype=np.float64)
            thresholds_y = np.asarray(getattr(model, "y_thresholds_", []), dtype=np.float64)
            if len(thresholds_x) >= 2:
                dx = float(thresholds_x[-1] - thresholds_x[0])
                dy = float(thresholds_y[-1] - thresholds_y[0]) if len(thresholds_y) >= 2 else float("nan")
                diagnostics["slope"] = float(dy / max(abs(dx), 1e-8)) if np.isfinite(dy) else float("nan")
            diagnostics["threshold_count"] = float(len(thresholds_x))
            diagnostics["output_min"] = _finite_stat(pred, "min")
            diagnostics["output_max"] = _finite_stat(pred, "max")
            return pred, diagnostics
        except Exception:
            return None
    return None


def _calibrate_prior(
    *,
    mode: str,
    fit_pred: np.ndarray,
    fit_y: np.ndarray,
    eval_pred: np.ndarray,
) -> np.ndarray | None:
    fit = _calibrate_prior_with_diagnostics(
        mode=mode,
        fit_pred=fit_pred,
        fit_y=fit_y,
        eval_pred=eval_pred,
    )
    return None if fit is None else fit[0]


def _weighted_calibrate_prior(
    *,
    mode: str,
    fit_pred: np.ndarray,
    fit_y: np.ndarray,
    eval_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray | None:
    x = np.asarray(fit_pred, dtype=np.float64).reshape(-1)
    y = np.asarray(fit_y, dtype=np.float64).reshape(-1)
    z = np.asarray(eval_pred, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if len(w) != len(x):
            return None
        keep &= np.isfinite(w) & (w > 0.0)
    else:
        w = None
    if int(np.sum(keep)) < 2 or not np.all(np.isfinite(z)):
        return None
    x = x[keep]
    y = y[keep]
    w_fit = w[keep] if w is not None else None
    mode = str(mode).strip().lower()
    if mode in ("linear", "affine"):
        if len(np.unique(x)) < 2:
            return None
        try:
            if w_fit is None:
                a, b = calibrate_linear(y, x)
            else:
                a, b = np.polyfit(x, y, deg=1, w=np.sqrt(np.clip(w_fit, 1e-12, None)))
                if not np.isfinite(a) or not np.isfinite(b):
                    return None
                a = float(np.clip(a, 0.1, 10.0))
                b = float(np.clip(b, -10.0 * np.std(y), 10.0 * np.std(y)))
            return apply_calibration(float(a), float(b), z)
        except Exception:
            return None
    if mode in ("isotonic", "iso"):
        if len(np.unique(x)) < 2:
            return None
        model = IsotonicRegression(out_of_bounds="clip")
        try:
            if w_fit is None:
                model.fit(x, y)
            else:
                model.fit(x, y, sample_weight=w_fit)
            return np.asarray(model.predict(z), dtype=np.float32)
        except Exception:
            return None
    return None


def _cp_similarity_source_mask_and_weights(
    ctx: CandidateBuildContext,
    aux_X_cp: np.ndarray,
    aux_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(len(aux_X_cp))
    if n_rows == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32)

    target_cp = _target_cp_ref_for_features(ctx.hyper_bundle[0] if isinstance(ctx.hyper_bundle, list) else ctx.hyper_bundle, ctx.X_train_cp)
    cp = np.asarray(aux_X_cp, dtype=np.float32)
    try:
        scaled = StandardScaler().fit_transform(np.vstack([cp, target_cp.reshape(1, -1)]))
        src_scaled = scaled[:-1]
        tgt_scaled = scaled[-1]
        dist = np.linalg.norm(src_scaled - tgt_scaled.reshape(1, -1), axis=1)
    except Exception:
        dist = np.linalg.norm(cp - target_cp.reshape(1, -1), axis=1)
    finite = np.isfinite(dist)
    if not np.any(finite):
        return np.zeros(n_rows, dtype=bool), np.zeros(n_rows, dtype=np.float32)

    denom = float(np.nanmedian(dist[finite]))
    if not np.isfinite(denom) or denom <= 1e-8:
        denom = float(np.nanmean(dist[finite])) if np.any(finite) else 1.0
    denom = max(denom, 1e-6)
    power = float(ctx.model_cfg.get("SHEET_PRIOR_CAL_SIMILARITY_POWER", 2.0) or 2.0)
    row_scores = np.exp(-np.power(dist / denom, max(power, 1e-6))).astype(np.float64)
    row_scores[~finite] = 0.0

    ids = np.asarray(aux_ids, dtype=object).reshape(-1) if aux_ids is not None and len(aux_ids) == n_rows else None
    if ids is None:
        return row_scores > 0.0, row_scores.astype(np.float32)

    target_id = str(ctx.target_dataset_id).zfill(4) if ctx.target_dataset_id is not None else ""
    include_target_aux = bool(ctx.model_cfg.get("SHEET_PRIOR_CAL_INCLUDE_TARGET_AUX", False))
    candidate_ids = sorted({str(raw).zfill(4) for raw in ids.tolist()})
    scored_ids: list[tuple[float, str]] = []
    for ds in candidate_ids:
        if target_id and ds == target_id and not include_target_aux:
            continue
        ds_mask = np.array([str(raw).zfill(4) == ds for raw in ids.tolist()], dtype=bool)
        if not np.any(ds_mask):
            continue
        score = float(np.nanmax(row_scores[ds_mask]))
        if np.isfinite(score) and score > 0.0:
            scored_ids.append((score, ds))
    top_k = int(ctx.model_cfg.get("SHEET_PRIOR_CAL_TOP_K_DATASETS", 8) or 8)
    if top_k > 0:
        keep_ids = {ds for _score, ds in sorted(scored_ids, reverse=True)[:top_k]}
    else:
        keep_ids = {ds for _score, ds in scored_ids}
    mask = np.array([str(raw).zfill(4) in keep_ids for raw in ids.tolist()], dtype=bool)
    weights = row_scores.astype(np.float32)
    weights[~mask] = 0.0
    if np.any(mask):
        weights[mask] *= _balanced_weights_for_ids(ids[mask])
    return mask, weights.astype(np.float32)


def _source_prior_raw_by_cp(bundle: Any, X_mol: np.ndarray, X_cp: np.ndarray, dataset_ids: np.ndarray | None) -> np.ndarray | None:
    X_mol = np.asarray(X_mol, dtype=np.float32)
    X_cp = np.asarray(X_cp, dtype=np.float32)
    if len(X_mol) != len(X_cp):
        return None
    out = np.full(len(X_mol), np.nan, dtype=np.float32)
    if len(X_mol) == 0:
        return out
    if dataset_ids is not None and len(dataset_ids) == len(X_mol):
        groups = [np.where(np.asarray(dataset_ids, dtype=object).reshape(-1) == raw)[0] for raw in sorted(set(dataset_ids.tolist()))]
    else:
        _, inv = np.unique(np.round(X_cp.astype(np.float64), 6), axis=0, return_inverse=True)
        groups = [np.where(inv == i)[0] for i in range(int(np.max(inv)) + 1)]
    for idx in groups:
        if len(idx) == 0:
            continue
        cp_ref = X_cp[idx[0]]
        ds_id = str(dataset_ids[idx[0]]).zfill(4) if dataset_ids is not None and len(dataset_ids) == len(X_mol) else None
        try:
            w0, b0 = head_prior(bundle, cp_ref)
            z = _target_embeddings(bundle, X_mol[idx], cp_ref, ds_id)
            out[idx] = (z @ w0 + b0).astype(np.float32)
        except Exception:
            continue
    if not np.all(np.isfinite(out)):
        return None
    return out


def _sheet_prior_fit_arrays(
    ctx: CandidateBuildContext,
    bundle: Any,
    *,
    include_target_val: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    aux_X_mol = getattr(bundle, "aux_X_mol", None)
    aux_X_cp = getattr(bundle, "aux_X_cp", None)
    aux_y = getattr(bundle, "aux_y", None)
    aux_ids = getattr(bundle, "aux_dataset_ids", None)
    aux_keys = getattr(bundle, "aux_mol_keys", None)
    if aux_X_mol is None or aux_X_cp is None or aux_y is None:
        return None
    aux_X_mol = np.asarray(aux_X_mol, dtype=np.float32)
    aux_X_cp = np.asarray(aux_X_cp, dtype=np.float32)
    aux_y = np.asarray(aux_y, dtype=np.float32).reshape(-1)
    aux_ids_arr = np.asarray(aux_ids, dtype=object).reshape(-1) if aux_ids is not None and len(aux_ids) == len(aux_y) else None
    if len(aux_y) < 20 or len(aux_X_mol) != len(aux_y) or len(aux_X_cp) != len(aux_y):
        return None

    source_mask, source_weight = _cp_similarity_source_mask_and_weights(ctx, aux_X_cp, aux_ids_arr)
    if not include_target_val:
        exclude_keys = _target_val_keys(ctx)
        if exclude_keys and aux_keys is not None and len(aux_keys) == len(aux_y):
            key_arr = np.asarray(aux_keys, dtype=object).reshape(-1)
            source_mask &= np.array([str(key) not in exclude_keys for key in key_arr.tolist()], dtype=bool)
            source_weight[~source_mask] = 0.0
    min_rows = int(ctx.model_cfg.get("SHEET_PRIOR_CAL_MIN_SOURCE_ROWS", 20) or 20)
    use_source = source_mask & np.isfinite(source_weight) & (source_weight > 0.0)
    if int(np.sum(use_source)) < min_rows:
        use_source = np.zeros_like(source_mask, dtype=bool)

    parts_x: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    parts_w: list[np.ndarray] = []
    if np.any(use_source):
        source_raw = _source_prior_raw_by_cp(
            bundle,
            aux_X_mol[use_source],
            aux_X_cp[use_source],
            aux_ids_arr[use_source] if aux_ids_arr is not None else None,
        )
        if source_raw is not None:
            src_w = source_weight[use_source].astype(np.float32)
            scale = float(ctx.model_cfg.get("SHEET_PRIOR_CAL_SOURCE_WEIGHT", 1.0) or 1.0)
            if np.any(src_w > 0):
                src_w = src_w / max(float(np.mean(src_w[src_w > 0])), 1e-12)
            parts_x.append(source_raw)
            parts_y.append(aux_y[use_source])
            parts_w.append((src_w * scale).astype(np.float32))

    cp_ref = _target_cp_ref(ctx, bundle)
    train_x = _target_embeddings(bundle, ctx.X_train_mol, cp_ref, ctx.target_dataset_id) @ head_prior(bundle, cp_ref)[0] + head_prior(bundle, cp_ref)[1]
    target_x_parts = [np.asarray(train_x, dtype=np.float32)]
    target_y_parts = [np.asarray(ctx.y_train, dtype=np.float32)]
    if include_target_val:
        val_x = _target_embeddings(bundle, ctx.X_val_mol, cp_ref, ctx.target_dataset_id) @ head_prior(bundle, cp_ref)[0] + head_prior(bundle, cp_ref)[1]
        target_x_parts.append(np.asarray(val_x, dtype=np.float32))
        target_y_parts.append(np.asarray(ctx.y_val_used, dtype=np.float32))
    target_x = np.concatenate(target_x_parts, axis=0)
    target_y = np.concatenate(target_y_parts, axis=0)
    target_weight = float(ctx.model_cfg.get("SHEET_PRIOR_CAL_TARGET_WEIGHT", 4.0) or 4.0)
    parts_x.append(target_x)
    parts_y.append(target_y)
    parts_w.append(np.full(len(target_y), target_weight, dtype=np.float32))
    fit_x = np.concatenate(parts_x, axis=0)
    fit_y = np.concatenate(parts_y, axis=0)
    fit_w = np.concatenate(parts_w, axis=0)
    keep = np.isfinite(fit_x) & np.isfinite(fit_y) & np.isfinite(fit_w) & (fit_w > 0.0)
    if int(np.sum(keep)) < 2:
        return None
    return fit_x[keep], fit_y[keep], fit_w[keep]


def _prior_raw_for_target(bundle: Any, X_mol: np.ndarray, cp_ref: np.ndarray, dataset_id: str | None) -> np.ndarray:
    w0, b0 = head_prior(bundle, cp_ref)
    return (_target_embeddings(bundle, X_mol, cp_ref, dataset_id) @ w0 + b0).astype(np.float32)


def _build_sheet_prior_calibration_candidates(ctx: CandidateBuildContext, bundles: list[Any]) -> list[CandidateOutput]:
    if len(ctx.y_train) < 2 or not bundles:
        return []
    modes = [str(x).strip().lower() for x in ctx.model_cfg.get("SHEET_PRIOR_CAL_MODES", ["linear", "isotonic"])]
    modes = [mode for mode in modes if mode]
    candidates: list[CandidateOutput] = []
    for mode in modes:
        val_preds: list[np.ndarray] = []
        test_preds: list[np.ndarray] = []
        for bundle in bundles:
            cp_ref = _target_cp_ref(ctx, bundle)
            fit_val = _sheet_prior_fit_arrays(ctx, bundle, include_target_val=False)
            fit_test = _sheet_prior_fit_arrays(ctx, bundle, include_target_val=True)
            if fit_val is None or fit_test is None:
                continue
            val_raw = _prior_raw_for_target(bundle, ctx.X_val_mol, cp_ref, ctx.target_dataset_id)
            test_raw = _prior_raw_for_target(bundle, ctx.X_test_mol, cp_ref, ctx.target_dataset_id)
            val_cal = _weighted_calibrate_prior(
                mode=mode,
                fit_pred=fit_val[0],
                fit_y=fit_val[1],
                eval_pred=val_raw,
                sample_weight=fit_val[2],
            )
            test_cal = _weighted_calibrate_prior(
                mode=mode,
                fit_pred=fit_test[0],
                fit_y=fit_test[1],
                eval_pred=test_raw,
                sample_weight=fit_test[2],
            )
            if val_cal is None or test_cal is None:
                continue
            val_preds.append(
                _inverse_target(val_cal, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
            )
            test_preds.append(
                _inverse_target(test_cal, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
            )
        if not val_preds:
            continue
        val_out = np.median(np.stack(val_preds, axis=0), axis=0).astype(np.float32)
        test_out = np.median(np.stack(test_preds, axis=0), axis=0).astype(np.float32)
        candidates.append(
            CandidateOutput(
                name=f"SHEET_PRIOR_CAL_{mode.upper()}(n={len(val_preds)})",
                val_pred=val_out,
                test_pred=test_out,
                val_metrics=compute_metrics(ctx.y_val_sec, val_out),
                model={"type": "sheet_prior_calibration", "mode": mode, "n": len(val_preds)},
            )
        )
    return candidates


def _build_hyper_emb_lgbm(ctx: CandidateBuildContext, bundle: Any) -> CandidateOutput:
    cp_ref = _target_cp_ref(ctx, bundle)
    Z_tr = _target_embeddings(bundle, ctx.X_train_mol, cp_ref, ctx.target_dataset_id)
    Z_va = _target_embeddings(bundle, ctx.X_val_mol, cp_ref, ctx.target_dataset_id)
    Z_te = _target_embeddings(bundle, ctx.X_test_mol, cp_ref, ctx.target_dataset_id)
    lgbm_params = ctx.model_cfg.get(
        "HYPER_EMB_LGBM",
        {
            "n_estimators": 2000,
            "num_leaves": 63,
            "learning_rate": 0.03,
            "objective": "mae",
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
        },
    )
    stop_rounds = int(ctx.model_cfg.get("HYPER_EMB_EARLY_STOPPING_ROUNDS", 50))
    z_cols = [f"z{i}" for i in range(Z_tr.shape[1])]
    Z_tr_df = pd.DataFrame(Z_tr, columns=z_cols)
    Z_va_df = pd.DataFrame(Z_va, columns=z_cols)
    Z_te_df = pd.DataFrame(Z_te, columns=z_cols)
    model = LGBMRegressor(random_state=ctx.seed + 7000, n_jobs=8, verbose=-1, **lgbm_params)
    callbacks = [early_stopping(stopping_rounds=stop_rounds, verbose=False)] if stop_rounds > 0 else None
    model.fit(
        Z_tr_df,
        ctx.y_train,
        sample_weight=np.full(len(ctx.y_train), float(ctx.target_weight), dtype=np.float32),
        eval_set=[(Z_va_df, ctx.y_val_used)],
        eval_metric="l1",
        callbacks=callbacks,
    )
    val_pred = _inverse_target(
        model.predict(Z_va_df),
        ctx.target_transform,
        ctx.target_inv_scale,
        ctx.target_t0_sec,
    )
    test_pred = _inverse_target(
        model.predict(Z_te_df),
        ctx.target_transform,
        ctx.target_inv_scale,
        ctx.target_t0_sec,
    )
    return CandidateOutput(
        name="HYPER_EMB_LGBM",
        val_pred=val_pred,
        test_pred=test_pred,
        val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
        model=model,
    )


def _balanced_weights_for_ids(ids: np.ndarray) -> np.ndarray:
    values = np.asarray(ids, dtype=object).reshape(-1)
    counts: dict[str, int] = {}
    for raw in values.tolist():
        key = str(raw).zfill(4)
        counts[key] = counts.get(key, 0) + 1
    weights = np.array([1.0 / max(counts.get(str(raw).zfill(4), 1), 1) for raw in values.tolist()], dtype=np.float32)
    mean = float(np.mean(weights)) if len(weights) else 1.0
    return (weights / max(mean, 1e-12)).astype(np.float32)


def _cp_similarity_weights(ctx: CandidateBuildContext, X_cp: np.ndarray) -> np.ndarray:
    cp = np.asarray(X_cp, dtype=np.float32)
    if cp.ndim != 2 or len(cp) == 0:
        return np.ones(len(cp), dtype=np.float32)
    if len(ctx.X_train_cp):
        target_cp = np.asarray(ctx.X_train_cp[0], dtype=np.float32).reshape(1, -1)
    else:
        target_cp = np.asarray(cp[0], dtype=np.float32).reshape(1, -1)
    try:
        scaled = StandardScaler().fit_transform(np.vstack([cp, target_cp]))
        src_scaled = scaled[:-1]
        tgt_scaled = scaled[-1]
        dist = np.linalg.norm(src_scaled - tgt_scaled.reshape(1, -1), axis=1)
    except Exception:
        dist = np.linalg.norm(cp - target_cp, axis=1)
    finite = np.isfinite(dist)
    if not np.any(finite):
        return np.ones(len(cp), dtype=np.float32)
    denom = float(np.nanmedian(dist[finite]))
    if not np.isfinite(denom) or denom <= 1e-8:
        denom = float(np.nanmean(dist[finite])) if np.any(finite) else 1.0
    denom = max(denom, 1e-6)
    power = float(ctx.model_cfg.get("SHEET_HEAD_CP_SIM_POWER", 2.0) or 2.0)
    min_weight = float(ctx.model_cfg.get("SHEET_HEAD_CP_SIM_MIN_WEIGHT", 0.05) or 0.05)
    raw = np.exp(-np.power(dist / denom, max(power, 1e-6))).astype(np.float32)
    raw[~finite] = 0.0
    weights = np.clip(raw, min_weight, 1.0).astype(np.float32)
    mean = float(np.mean(weights)) if len(weights) else 1.0
    return (weights / max(mean, 1e-12)).astype(np.float32)


def _target_val_keys(ctx: CandidateBuildContext) -> set[str]:
    keys = ctx.target_mol_keys
    if keys is None:
        return set()
    ordered = np.asarray(keys, dtype=object).reshape(-1)
    start = len(ctx.X_train_mol)
    stop = start + len(ctx.X_val_mol)
    return {str(x) for x in ordered[start:stop].tolist() if str(x)}


def _target_fit_keys(ctx: CandidateBuildContext) -> set[str]:
    keys = ctx.target_mol_keys
    if keys is None:
        return set()
    ordered = np.asarray(keys, dtype=object).reshape(-1)
    stop = len(ctx.X_train_mol) + len(ctx.X_val_mol)
    return {str(x) for x in ordered[:stop].tolist() if str(x)}


def _full_mol_dim(group_sizes: dict[str, int]) -> int:
    return int(
        group_sizes.get("descriptor", 0)
        + group_sizes.get("fingerprint", 0)
        + group_sizes.get("mol_text", 0)
        + group_sizes.get("mol_seq", 0)
    )


def _memory_mol_block(X: np.ndarray, group_sizes: dict[str, int], mode: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mode = str(mode or "full").strip().lower()
    desc = int(group_sizes.get("descriptor", 0))
    fp = int(group_sizes.get("fingerprint", 0))
    if mode in ("fingerprint", "fp", "fingerprints") and fp > 0 and X.shape[1] >= desc + fp:
        return X[:, desc : desc + fp]
    mol_dim = _full_mol_dim(group_sizes)
    return X[:, :mol_dim]


def _row_l2_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    denom = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(denom, 1e-6)).astype(np.float32)


def _attention_raw_predict(
    *,
    fit_mol: np.ndarray,
    fit_cp: np.ndarray,
    fit_y: np.ndarray,
    eval_mol: np.ndarray,
    eval_cp: np.ndarray,
    top_k: int,
    mol_power: float,
    cp_power: float,
    cp_weight: float,
) -> np.ndarray:
    fit_mol = np.asarray(fit_mol, dtype=np.float32)
    eval_mol = np.asarray(eval_mol, dtype=np.float32)
    fit_cp = np.asarray(fit_cp, dtype=np.float32)
    eval_cp = np.asarray(eval_cp, dtype=np.float32)
    fit_y = np.asarray(fit_y, dtype=np.float32).reshape(-1)
    if len(fit_y) == 0 or fit_mol.shape[0] != len(fit_y):
        return np.zeros(len(eval_mol), dtype=np.float32)

    mol_mean = fit_mol.mean(axis=0, keepdims=True)
    mol_std = fit_mol.std(axis=0, keepdims=True)
    mol_std = np.where(mol_std < 1e-6, 1.0, mol_std).astype(np.float32)
    fit_m = _row_l2_normalize((fit_mol - mol_mean) / mol_std)
    eval_m = _row_l2_normalize((eval_mol - mol_mean) / mol_std)
    mol_sim = np.maximum(eval_m @ fit_m.T, 0.0)
    if mol_power != 1.0:
        mol_sim = np.power(mol_sim, max(float(mol_power), 1e-6))

    scores = mol_sim.astype(np.float64)
    if fit_cp.shape[1] and fit_cp.shape == (len(fit_y), eval_cp.shape[1]) and cp_weight > 0.0:
        cp_mean = fit_cp.mean(axis=0, keepdims=True)
        cp_std = fit_cp.std(axis=0, keepdims=True)
        cp_std = np.where(cp_std < 1e-6, 1.0, cp_std).astype(np.float32)
        fit_c = (fit_cp - cp_mean) / cp_std
        eval_c = (eval_cp - cp_mean) / cp_std
        diff = eval_c[:, None, :] - fit_c[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        finite = np.isfinite(dist)
        denom = float(np.nanmedian(dist[finite])) if np.any(finite) else 1.0
        denom = max(denom, 1e-6)
        cp_sim = np.exp(-np.power(dist / denom, max(float(cp_power), 1e-6)))
        scores *= np.power(np.maximum(cp_sim, 1e-12), float(cp_weight))

    out = np.zeros(len(eval_mol), dtype=np.float32)
    k = int(max(1, min(int(top_k), len(fit_y))))
    global_median = float(np.nanmedian(fit_y)) if len(fit_y) else 0.0
    for row in range(len(eval_mol)):
        s = scores[row]
        if not np.any(np.isfinite(s)) or float(np.nansum(s)) <= 1e-12:
            out[row] = global_median
            continue
        idx = np.argpartition(-s, kth=k - 1)[:k]
        w = np.asarray(s[idx], dtype=np.float64)
        w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
        if float(np.sum(w)) <= 1e-12:
            out[row] = global_median
        else:
            out[row] = float(np.dot(w, fit_y[idx]) / np.sum(w))
    return out


def _calibrate_memory_raw(fit_pred: np.ndarray, fit_y: np.ndarray, eval_pred: np.ndarray) -> np.ndarray:
    x = np.asarray(fit_pred, dtype=np.float64).reshape(-1)
    y = np.asarray(fit_y, dtype=np.float64).reshape(-1)
    z = np.asarray(eval_pred, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(keep)) < 2 or len(np.unique(x[keep])) < 2:
        return z.astype(np.float32)
    a, b = calibrate_linear(y[keep], x[keep])
    return apply_calibration(a, b, z).astype(np.float32)


def _build_sheet_memory_attention(ctx: CandidateBuildContext, bundle: Any) -> CandidateOutput | None:
    aux_X_full = getattr(bundle, "aux_X_full", None)
    aux_X_cp = getattr(bundle, "aux_X_cp", None)
    aux_y = getattr(bundle, "aux_y", None)
    aux_keys = getattr(bundle, "aux_mol_keys", None)
    if aux_X_full is None or aux_X_cp is None or aux_y is None:
        return None
    aux_X_full = np.asarray(aux_X_full, dtype=np.float32)
    aux_X_cp = np.asarray(aux_X_cp, dtype=np.float32)
    aux_y = np.asarray(aux_y, dtype=np.float32).reshape(-1)
    mol_dim = _full_mol_dim(ctx.group_sizes)
    if mol_dim <= 0 or aux_X_full.shape[1] < mol_dim or ctx.X_val.shape[1] < mol_dim or ctx.X_test.shape[1] < mol_dim:
        return None
    if len(aux_y) < 20 or len(aux_X_full) != len(aux_y) or len(aux_X_cp) != len(aux_y):
        return None

    key_arr = np.asarray(aux_keys, dtype=object).reshape(-1) if aux_keys is not None and len(aux_keys) == len(aux_y) else None
    fit_keys = _target_fit_keys(ctx)
    no_fit_leak = np.ones(len(aux_y), dtype=bool)
    if fit_keys and key_arr is not None:
        no_fit_leak &= np.array([str(key) not in fit_keys for key in key_arr.tolist()], dtype=bool)
    if int(np.sum(no_fit_leak)) < 20:
        return None

    top_k = int(ctx.model_cfg.get("SHEET_MEMORY_TOP_K", 12) or 12)
    mol_power = float(ctx.model_cfg.get("SHEET_MEMORY_MOL_POWER", 4.0) or 4.0)
    cp_power = float(ctx.model_cfg.get("SHEET_MEMORY_CP_POWER", 2.0) or 2.0)
    cp_weight = float(ctx.model_cfg.get("SHEET_MEMORY_CP_WEIGHT", 1.0) or 1.0)
    feature_mode = str(ctx.model_cfg.get("SHEET_MEMORY_FEATURES", "full") or "full")
    calibrate = bool(ctx.model_cfg.get("SHEET_MEMORY_CALIBRATE", True))
    aux_mol_block = _memory_mol_block(aux_X_full, ctx.group_sizes, feature_mode)
    train_mol_block = _memory_mol_block(ctx.X_train, ctx.group_sizes, feature_mode)
    val_mol_block = _memory_mol_block(ctx.X_val, ctx.group_sizes, feature_mode)
    test_mol_block = _memory_mol_block(ctx.X_test, ctx.group_sizes, feature_mode)

    train_raw = _attention_raw_predict(
        fit_mol=aux_mol_block[no_fit_leak],
        fit_cp=aux_X_cp[no_fit_leak],
        fit_y=aux_y[no_fit_leak],
        eval_mol=train_mol_block,
        eval_cp=ctx.X_train_cp,
        top_k=top_k,
        mol_power=mol_power,
        cp_power=cp_power,
        cp_weight=cp_weight,
    )
    val_raw = _attention_raw_predict(
        fit_mol=aux_mol_block[no_fit_leak],
        fit_cp=aux_X_cp[no_fit_leak],
        fit_y=aux_y[no_fit_leak],
        eval_mol=val_mol_block,
        eval_cp=ctx.X_val_cp if ctx.X_val_cp is not None else ctx.X_train_cp[:0],
        top_k=top_k,
        mol_power=mol_power,
        cp_power=cp_power,
        cp_weight=cp_weight,
    )
    if calibrate:
        val_raw = _calibrate_memory_raw(train_raw, ctx.y_train, val_raw)

    final_mask = np.ones(len(aux_y), dtype=bool)
    if key_arr is not None:
        val_keys = _target_val_keys(ctx)
        if val_keys:
            final_mask &= np.array([str(key) not in val_keys for key in key_arr.tolist()], dtype=bool)
    if int(np.sum(final_mask)) < 20:
        final_mask = np.ones(len(aux_y), dtype=bool)
    train_val_raw = _attention_raw_predict(
        fit_mol=aux_mol_block[final_mask],
        fit_cp=aux_X_cp[final_mask],
        fit_y=aux_y[final_mask],
        eval_mol=np.concatenate([train_mol_block, val_mol_block], axis=0),
        eval_cp=np.concatenate([ctx.X_train_cp, ctx.X_val_cp if ctx.X_val_cp is not None else ctx.X_train_cp[:0]], axis=0),
        top_k=top_k,
        mol_power=mol_power,
        cp_power=cp_power,
        cp_weight=cp_weight,
    )
    test_raw = _attention_raw_predict(
        fit_mol=aux_mol_block[final_mask],
        fit_cp=aux_X_cp[final_mask],
        fit_y=aux_y[final_mask],
        eval_mol=test_mol_block,
        eval_cp=ctx.X_test_cp if ctx.X_test_cp is not None else ctx.X_train_cp[:0],
        top_k=top_k,
        mol_power=mol_power,
        cp_power=cp_power,
        cp_weight=cp_weight,
    )
    if calibrate:
        test_raw = _calibrate_memory_raw(
            train_val_raw,
            np.concatenate([ctx.y_train, ctx.y_val_used], axis=0),
            test_raw,
        )
    val_pred = _inverse_target(val_raw, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    test_pred = _inverse_target(test_raw, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    return CandidateOutput(
        name="SHEET_MEMORY_ATTENTION",
        val_pred=val_pred,
        test_pred=test_pred,
        val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
        model={"type": "sheet_memory_attention", "top_k": top_k, "features": feature_mode},
    )


def _sheet_head_features(bundle: Any, X_mol: np.ndarray, X_cp: np.ndarray, dataset_id: str | None = None) -> pd.DataFrame:
    cp_ref = _target_cp_ref_for_features(bundle, X_cp)
    Z = _target_embeddings(bundle, X_mol, cp_ref, dataset_id)
    cp = np.asarray(X_cp, dtype=np.float32)
    data = np.concatenate([Z, cp], axis=1).astype(np.float32)
    cols = [f"z{i}" for i in range(Z.shape[1])] + [f"cp{i}" for i in range(cp.shape[1])]
    return pd.DataFrame(data, columns=cols)


def _target_cp_ref_for_features(bundle: Any, X_cp: np.ndarray) -> np.ndarray:
    cp = np.asarray(X_cp, dtype=np.float32)
    if cp.ndim == 2 and len(cp):
        return cp[0]
    return np.zeros_like(np.asarray(bundle.cp_mean, dtype=np.float32))


def _build_sheet_emb_lgbm(ctx: CandidateBuildContext, bundle: Any, *, use_seconds: bool = False) -> CandidateOutput | None:
    aux_X_mol = getattr(bundle, "aux_X_mol", None)
    aux_X_cp = getattr(bundle, "aux_X_cp", None)
    aux_y = getattr(bundle, "aux_y_sec", None) if use_seconds else getattr(bundle, "aux_y", None)
    aux_ids = getattr(bundle, "aux_dataset_ids", None)
    aux_keys = getattr(bundle, "aux_mol_keys", None)
    if aux_X_mol is None or aux_X_cp is None or aux_y is None:
        return None
    aux_X_mol = np.asarray(aux_X_mol, dtype=np.float32)
    aux_X_cp = np.asarray(aux_X_cp, dtype=np.float32)
    aux_y = np.asarray(aux_y, dtype=np.float32).reshape(-1)
    if len(aux_y) < 20 or len(aux_X_mol) != len(aux_y) or len(aux_X_cp) != len(aux_y):
        return None

    exclude_keys = _target_val_keys(ctx)
    train_mask = np.ones(len(aux_y), dtype=bool)
    if exclude_keys and aux_keys is not None and len(aux_keys) == len(aux_y):
        key_arr = np.asarray(aux_keys, dtype=object).reshape(-1)
        train_mask &= np.array([str(key) not in exclude_keys for key in key_arr.tolist()], dtype=bool)
    if int(np.sum(train_mask)) < 20:
        return None

    cfg_key = "SHEET_EMB_LGBM_SEC" if use_seconds else "SHEET_EMB_LGBM"
    params = ctx.model_cfg.get(
        cfg_key,
        {
            "n_estimators": 2500,
            "num_leaves": 63,
            "learning_rate": 0.025,
            "objective": "mae",
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_child_samples": 8,
        },
    )
    stop_rounds = int(ctx.model_cfg.get("SHEET_EMB_SEC_EARLY_STOPPING_ROUNDS" if use_seconds else "SHEET_EMB_EARLY_STOPPING_ROUNDS", 80))
    if ctx.X_val_cp is None or ctx.X_test_cp is None:
        return None
    X_fit = _sheet_head_features(bundle, aux_X_mol[train_mask], aux_X_cp[train_mask])
    X_val = _sheet_head_features(bundle, ctx.X_val_mol, ctx.X_val_cp, ctx.target_dataset_id)
    X_test = _sheet_head_features(bundle, ctx.X_test_mol, ctx.X_test_cp, ctx.target_dataset_id)
    sample_weight = None
    if aux_ids is not None and len(aux_ids) == len(aux_y):
        sample_weight = _balanced_weights_for_ids(np.asarray(aux_ids, dtype=object)[train_mask])
    if bool(ctx.model_cfg.get("SHEET_HEAD_CP_SIM_WEIGHT", False)):
        cp_weight = _cp_similarity_weights(ctx, aux_X_cp[train_mask])
        sample_weight = cp_weight if sample_weight is None else (sample_weight * cp_weight).astype(np.float32)
    model = LGBMRegressor(random_state=ctx.seed + 9100, n_jobs=8, verbose=-1, **params)
    callbacks = [early_stopping(stopping_rounds=stop_rounds, verbose=False)] if stop_rounds > 0 else None
    model.fit(
        X_fit,
        aux_y[train_mask],
        sample_weight=sample_weight,
        eval_set=[(X_val, ctx.y_val_sec if use_seconds else ctx.y_val_used)],
        eval_metric="l1",
        callbacks=callbacks,
    )
    best_iter = int(getattr(model, "best_iteration_", 0)) or None
    val_raw = model.predict(X_val, num_iteration=best_iter)

    X_final = _sheet_head_features(bundle, aux_X_mol, aux_X_cp)
    final_params = dict(params)
    if best_iter is not None:
        final_params["n_estimators"] = max(1, int(best_iter))
    final_weight = _balanced_weights_for_ids(np.asarray(aux_ids, dtype=object)) if aux_ids is not None and len(aux_ids) == len(aux_y) else None
    if bool(ctx.model_cfg.get("SHEET_HEAD_CP_SIM_WEIGHT", False)):
        final_cp_weight = _cp_similarity_weights(ctx, aux_X_cp)
        final_weight = final_cp_weight if final_weight is None else (final_weight * final_cp_weight).astype(np.float32)
    final_model = LGBMRegressor(random_state=ctx.seed + 9101, n_jobs=8, verbose=-1, **final_params)
    final_model.fit(X_final, aux_y, sample_weight=final_weight)
    test_raw = final_model.predict(X_test)
    if use_seconds:
        val_pred = np.asarray(val_raw, dtype=np.float32)
        test_pred = np.asarray(test_raw, dtype=np.float32)
    else:
        val_pred = _inverse_target(val_raw, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
        test_pred = _inverse_target(test_raw, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    return CandidateOutput(
        name="SHEET_EMB_LGBM_SEC" if use_seconds else "SHEET_EMB_LGBM",
        val_pred=val_pred,
        test_pred=test_pred,
        val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
        model=final_model,
    )


def _full_head_frame(X: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(X, dtype=np.float32)
    return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])


def _prior_raw_for_rows(
    bundle: Any,
    X_mol: np.ndarray,
    X_cp: np.ndarray,
    dataset_ids: np.ndarray | None = None,
) -> np.ndarray:
    x_mol = np.asarray(X_mol, dtype=np.float32)
    x_cp = np.asarray(X_cp, dtype=np.float32)
    if len(x_mol) == 0:
        return np.empty((0,), dtype=np.float32)
    if x_cp.ndim != 2 or len(x_cp) != len(x_mol):
        cp_ref = _target_cp_ref_for_features(bundle, x_cp)
        w0, b0 = head_prior(bundle, cp_ref)
        z = _target_embeddings(bundle, x_mol, cp_ref, None)
        return (z @ w0 + b0).astype(np.float32)

    out = np.empty(len(x_mol), dtype=np.float32)
    ids = np.asarray(dataset_ids, dtype=object).reshape(-1) if dataset_ids is not None and len(dataset_ids) == len(x_mol) else None
    rounded_cp = np.round(x_cp.astype(np.float64), decimals=6)
    _, inverse = np.unique(rounded_cp, axis=0, return_inverse=True)
    for group_id in np.unique(inverse):
        mask = inverse == group_id
        cp_ref = x_cp[int(np.flatnonzero(mask)[0])]
        dataset_id = str(ids[int(np.flatnonzero(mask)[0])]).zfill(4) if ids is not None else None
        z = _target_embeddings(bundle, x_mol[mask], cp_ref, dataset_id)
        w0, b0 = head_prior(bundle, cp_ref)
        out[mask] = (z @ w0 + b0).astype(np.float32)
    return out


def _residual_head_frame(
    bundle: Any,
    X_mol: np.ndarray,
    X_cp: np.ndarray,
    prior_raw: np.ndarray,
    *,
    X_full: np.ndarray | None = None,
    dataset_ids: np.ndarray | None = None,
    dataset_id: str | None = None,
) -> pd.DataFrame:
    prior_col = np.asarray(prior_raw, dtype=np.float32).reshape(-1, 1)
    if X_full is not None:
        arr = np.concatenate([np.asarray(X_full, dtype=np.float32), prior_col], axis=1)
        cols = [f"f{i}" for i in range(arr.shape[1] - 1)] + ["prior_raw"]
        return pd.DataFrame(arr, columns=cols)

    x_mol = np.asarray(X_mol, dtype=np.float32)
    x_cp = np.asarray(X_cp, dtype=np.float32)
    if dataset_ids is not None and len(dataset_ids) == len(x_mol):
        ids = np.asarray(dataset_ids, dtype=object).reshape(-1)
        rounded_cp = np.round(x_cp.astype(np.float64), decimals=6)
        _, inverse = np.unique(rounded_cp, axis=0, return_inverse=True)
        z = np.empty((len(x_mol), int(getattr(bundle.model, "embed_dim", 0))), dtype=np.float32)
        for group_id in np.unique(inverse):
            mask = inverse == group_id
            cp_ref = x_cp[int(np.flatnonzero(mask)[0])]
            ds_id = str(ids[int(np.flatnonzero(mask)[0])]).zfill(4)
            z[mask] = _target_embeddings(bundle, x_mol[mask], cp_ref, ds_id)
        cp = x_cp
    else:
        frame = _sheet_head_features(bundle, x_mol, x_cp, dataset_id)
        z_cols = [col for col in frame.columns if col.startswith("z")]
        cp_cols = [col for col in frame.columns if col.startswith("cp")]
        z = frame[z_cols].to_numpy(dtype=np.float32)
        cp = frame[cp_cols].to_numpy(dtype=np.float32)
    arr = np.concatenate([z, cp, prior_col], axis=1).astype(np.float32)
    cols = [f"z{i}" for i in range(z.shape[1])] + [f"cp{i}" for i in range(cp.shape[1])] + ["prior_raw"]
    return pd.DataFrame(arr, columns=cols)


def _build_sheet_residual_lgbm(ctx: CandidateBuildContext, bundle: Any, *, use_full: bool = False) -> CandidateOutput | None:
    aux_X_mol = getattr(bundle, "aux_X_mol", None)
    aux_X_cp = getattr(bundle, "aux_X_cp", None)
    aux_y = getattr(bundle, "aux_y", None)
    aux_X_full = getattr(bundle, "aux_X_full", None) if use_full else None
    aux_ids = getattr(bundle, "aux_dataset_ids", None)
    aux_keys = getattr(bundle, "aux_mol_keys", None)
    if aux_X_mol is None or aux_X_cp is None or aux_y is None:
        return None
    aux_X_mol = np.asarray(aux_X_mol, dtype=np.float32)
    aux_X_cp = np.asarray(aux_X_cp, dtype=np.float32)
    aux_y = np.asarray(aux_y, dtype=np.float32).reshape(-1)
    if len(aux_y) < 20 or len(aux_X_mol) != len(aux_y) or len(aux_X_cp) != len(aux_y):
        return None
    if use_full:
        if aux_X_full is None:
            return None
        aux_X_full = np.asarray(aux_X_full, dtype=np.float32)
        if len(aux_X_full) != len(aux_y) or aux_X_full.shape[1] != ctx.X_val.shape[1]:
            return None
    exclude_keys = _target_val_keys(ctx)
    train_mask = np.ones(len(aux_y), dtype=bool)
    if exclude_keys and aux_keys is not None and len(aux_keys) == len(aux_y):
        key_arr = np.asarray(aux_keys, dtype=object).reshape(-1)
        train_mask &= np.array([str(key) not in exclude_keys for key in key_arr.tolist()], dtype=bool)
    if int(np.sum(train_mask)) < 20:
        return None

    cfg_key = "SHEET_FULL_RESID_LGBM" if use_full else "SHEET_RESID_LGBM"
    params = ctx.model_cfg.get(
        cfg_key,
        {
            "n_estimators": 2500,
            "num_leaves": 63,
            "learning_rate": 0.025,
            "objective": "mae",
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_child_samples": 8,
        },
    )
    stop_rounds = int(ctx.model_cfg.get("SHEET_RESID_EARLY_STOPPING_ROUNDS", 80))
    if ctx.X_val_cp is None or ctx.X_test_cp is None:
        return None

    ids_all = np.asarray(aux_ids, dtype=object) if aux_ids is not None and len(aux_ids) == len(aux_y) else None
    prior_aux = _prior_raw_for_rows(bundle, aux_X_mol, aux_X_cp, ids_all)
    prior_val = _prior_raw_for_rows(bundle, ctx.X_val_mol, ctx.X_val_cp, np.array([ctx.target_dataset_id] * len(ctx.X_val_mol), dtype=object))
    prior_test = _prior_raw_for_rows(bundle, ctx.X_test_mol, ctx.X_test_cp, np.array([ctx.target_dataset_id] * len(ctx.X_test_mol), dtype=object))
    resid_y = (aux_y - prior_aux).astype(np.float32)
    resid_val_y = (ctx.y_val_used - prior_val).astype(np.float32)

    X_fit = _residual_head_frame(
        bundle,
        aux_X_mol[train_mask],
        aux_X_cp[train_mask],
        prior_aux[train_mask],
        X_full=aux_X_full[train_mask] if use_full and aux_X_full is not None else None,
        dataset_ids=ids_all[train_mask] if ids_all is not None else None,
    )
    X_val = _residual_head_frame(
        bundle,
        ctx.X_val_mol,
        ctx.X_val_cp,
        prior_val,
        X_full=ctx.X_val if use_full else None,
        dataset_id=ctx.target_dataset_id,
    )
    X_test = _residual_head_frame(
        bundle,
        ctx.X_test_mol,
        ctx.X_test_cp,
        prior_test,
        X_full=ctx.X_test if use_full else None,
        dataset_id=ctx.target_dataset_id,
    )
    sample_weight = None
    if ids_all is not None:
        sample_weight = _balanced_weights_for_ids(ids_all[train_mask])
    if bool(ctx.model_cfg.get("SHEET_HEAD_CP_SIM_WEIGHT", False)):
        cp_weight = _cp_similarity_weights(ctx, aux_X_cp[train_mask])
        sample_weight = cp_weight if sample_weight is None else (sample_weight * cp_weight).astype(np.float32)

    model = LGBMRegressor(random_state=ctx.seed + (9500 if use_full else 9400), n_jobs=8, verbose=-1, **params)
    callbacks = [early_stopping(stopping_rounds=stop_rounds, verbose=False)] if stop_rounds > 0 else None
    model.fit(
        X_fit,
        resid_y[train_mask],
        sample_weight=sample_weight,
        eval_set=[(X_val, resid_val_y)],
        eval_metric="l1",
        callbacks=callbacks,
    )
    best_iter = int(getattr(model, "best_iteration_", 0)) or None
    val_raw = prior_val + np.asarray(model.predict(X_val, num_iteration=best_iter), dtype=np.float32)

    final_params = dict(params)
    if best_iter is not None:
        final_params["n_estimators"] = max(1, int(best_iter))
    final_weight = _balanced_weights_for_ids(ids_all) if ids_all is not None else None
    if bool(ctx.model_cfg.get("SHEET_HEAD_CP_SIM_WEIGHT", False)):
        final_cp_weight = _cp_similarity_weights(ctx, aux_X_cp)
        final_weight = final_cp_weight if final_weight is None else (final_weight * final_cp_weight).astype(np.float32)
    X_final = _residual_head_frame(
        bundle,
        aux_X_mol,
        aux_X_cp,
        prior_aux,
        X_full=aux_X_full if use_full and aux_X_full is not None else None,
        dataset_ids=ids_all,
    )
    final_model = LGBMRegressor(random_state=ctx.seed + (9501 if use_full else 9401), n_jobs=8, verbose=-1, **final_params)
    final_model.fit(X_final, resid_y, sample_weight=final_weight)
    test_raw = prior_test + np.asarray(final_model.predict(X_test), dtype=np.float32)
    val_pred = _inverse_target(val_raw, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    test_pred = _inverse_target(test_raw, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    return CandidateOutput(
        name="SHEET_FULL_RESID_LGBM" if use_full else "SHEET_RESID_LGBM",
        val_pred=val_pred,
        test_pred=test_pred,
        val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
        model=final_model,
    )


def _build_sheet_full_lgbm(ctx: CandidateBuildContext, bundle: Any, *, use_seconds: bool = False) -> CandidateOutput | None:
    aux_X = getattr(bundle, "aux_X_full", None)
    aux_y = getattr(bundle, "aux_y_sec", None) if use_seconds else getattr(bundle, "aux_y", None)
    aux_ids = getattr(bundle, "aux_dataset_ids", None)
    aux_keys = getattr(bundle, "aux_mol_keys", None)
    if aux_X is None or aux_y is None:
        return None
    aux_X = np.asarray(aux_X, dtype=np.float32)
    aux_y = np.asarray(aux_y, dtype=np.float32).reshape(-1)
    if len(aux_y) < 20 or len(aux_X) != len(aux_y) or aux_X.shape[1] != ctx.X_val.shape[1]:
        return None
    exclude_keys = _target_val_keys(ctx)
    train_mask = np.ones(len(aux_y), dtype=bool)
    if exclude_keys and aux_keys is not None and len(aux_keys) == len(aux_y):
        key_arr = np.asarray(aux_keys, dtype=object).reshape(-1)
        train_mask &= np.array([str(key) not in exclude_keys for key in key_arr.tolist()], dtype=bool)
    if int(np.sum(train_mask)) < 20:
        return None
    cfg_key = "SHEET_FULL_LGBM_SEC" if use_seconds else "SHEET_FULL_LGBM"
    params = ctx.model_cfg.get(
        cfg_key,
        {
            "n_estimators": 3000,
            "num_leaves": 63,
            "learning_rate": 0.025,
            "objective": "mae",
            "feature_fraction": 0.85,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_child_samples": 8,
        },
    )
    stop_rounds = int(ctx.model_cfg.get("SHEET_FULL_SEC_EARLY_STOPPING_ROUNDS" if use_seconds else "SHEET_FULL_EARLY_STOPPING_ROUNDS", 80))
    X_fit = _full_head_frame(aux_X[train_mask])
    X_val = _full_head_frame(ctx.X_val)
    X_test = _full_head_frame(ctx.X_test)
    sample_weight = None
    if aux_ids is not None and len(aux_ids) == len(aux_y):
        sample_weight = _balanced_weights_for_ids(np.asarray(aux_ids, dtype=object)[train_mask])
    if bool(ctx.model_cfg.get("SHEET_HEAD_CP_SIM_WEIGHT", False)):
        cp_all = getattr(bundle, "aux_X_cp", None)
        if cp_all is not None and len(cp_all) == len(aux_y):
            cp_weight = _cp_similarity_weights(ctx, np.asarray(cp_all, dtype=np.float32)[train_mask])
            sample_weight = cp_weight if sample_weight is None else (sample_weight * cp_weight).astype(np.float32)
    model = LGBMRegressor(random_state=ctx.seed + 9300, n_jobs=8, verbose=-1, **params)
    callbacks = [early_stopping(stopping_rounds=stop_rounds, verbose=False)] if stop_rounds > 0 else None
    model.fit(
        X_fit,
        aux_y[train_mask],
        sample_weight=sample_weight,
        eval_set=[(X_val, ctx.y_val_sec if use_seconds else ctx.y_val_used)],
        eval_metric="l1",
        callbacks=callbacks,
    )
    best_iter = int(getattr(model, "best_iteration_", 0)) or None
    val_raw = model.predict(X_val, num_iteration=best_iter)
    final_params = dict(params)
    if best_iter is not None:
        final_params["n_estimators"] = max(1, int(best_iter))
    final_weight = _balanced_weights_for_ids(np.asarray(aux_ids, dtype=object)) if aux_ids is not None and len(aux_ids) == len(aux_y) else None
    if bool(ctx.model_cfg.get("SHEET_HEAD_CP_SIM_WEIGHT", False)):
        cp_all = getattr(bundle, "aux_X_cp", None)
        if cp_all is not None and len(cp_all) == len(aux_y):
            final_cp_weight = _cp_similarity_weights(ctx, np.asarray(cp_all, dtype=np.float32))
            final_weight = final_cp_weight if final_weight is None else (final_weight * final_cp_weight).astype(np.float32)
    final_model = LGBMRegressor(random_state=ctx.seed + 9301, n_jobs=8, verbose=-1, **final_params)
    final_model.fit(_full_head_frame(aux_X), aux_y, sample_weight=final_weight)
    test_raw = final_model.predict(X_test)
    if use_seconds:
        val_pred = np.asarray(val_raw, dtype=np.float32)
        test_pred = np.asarray(test_raw, dtype=np.float32)
    else:
        val_pred = _inverse_target(val_raw, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
        test_pred = _inverse_target(test_raw, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    return CandidateOutput(
        name="SHEET_FULL_LGBM_SEC" if use_seconds else "SHEET_FULL_LGBM",
        val_pred=val_pred,
        test_pred=test_pred,
        val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
        model=final_model,
    )


def _build_hyper_emb_adapter(ctx: CandidateBuildContext, bundle: Any) -> CandidateOutput | None:
    adapter_cfg = ctx.model_cfg.get(
        "HYPER_EMB_ADAPTER",
        {
            "hidden": 96,
            "bottleneck": 24,
            "dropout": 0.05,
            "lr": 0.002,
            "weight_decay": 1e-4,
            "epochs": 300,
            "patience": 30,
            "batch_size": 64,
            "loss": "smooth_l1",
            "beta": 0.5,
            "refit_on_train_val": True,
            "min_train_rows": 12,
        },
    )
    min_train_rows = int(adapter_cfg.get("min_train_rows", 12))
    if len(ctx.y_train) < min_train_rows or len(ctx.y_val_used) == 0:
        return None

    cp_ref = _target_cp_ref(ctx, bundle)
    Z_tr = _target_embeddings(bundle, ctx.X_train_mol, cp_ref, ctx.target_dataset_id)
    Z_va = _target_embeddings(bundle, ctx.X_val_mol, cp_ref, ctx.target_dataset_id)
    Z_te = _target_embeddings(bundle, ctx.X_test_mol, cp_ref, ctx.target_dataset_id)

    fit = _fit_embedding_adapter(
        Z_train=Z_tr,
        y_train=ctx.y_train,
        Z_val=Z_va,
        y_val=ctx.y_val_used,
        Z_test=Z_te,
        cfg=adapter_cfg,
        seed=int(ctx.seed) + 8100,
        refit_on_train_val=bool(adapter_cfg.get("refit_on_train_val", True)),
    )
    if fit is None:
        return None
    val_raw, test_raw, model_info = fit
    val_pred = _inverse_target(val_raw, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    test_pred = _inverse_target(test_raw, ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
    return CandidateOutput(
        name="HYPER_EMB_ADAPTER",
        val_pred=val_pred,
        test_pred=test_pred,
        val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
        model=model_info,
    )


def _fit_embedding_adapter(
    *,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    Z_test: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
    refit_on_train_val: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit_scalers(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, StandardScaler]:
        xs = StandardScaler().fit(np.asarray(X, dtype=np.float32))
        ys = StandardScaler().fit(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        return xs, ys

    def _prepare(xs: StandardScaler, ys: StandardScaler, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        Xs = xs.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        if y is None:
            return Xs, None
        ys_out = ys.transform(np.asarray(y, dtype=np.float32).reshape(-1, 1)).ravel().astype(np.float32)
        return Xs, ys_out

    def _train_once(
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_watch: np.ndarray | None,
        y_watch: np.ndarray | None,
        max_epochs: int,
        patience: int,
        seed_offset: int,
    ) -> tuple[_EmbeddingAdapter, int] | None:
        if len(X_fit) == 0:
            return None
        torch.manual_seed(int(seed) + int(seed_offset))
        model = _EmbeddingAdapter(
            in_dim=X_fit.shape[1],
            hidden=int(cfg.get("hidden", 96)),
            bottleneck=int(cfg.get("bottleneck", 24)),
            dropout=float(cfg.get("dropout", 0.05)),
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 0.002)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
        loss_name = str(cfg.get("loss", "smooth_l1")).strip().lower()
        if loss_name in ("l1", "mae"):
            loss_fn: nn.Module = nn.L1Loss()
        else:
            loss_fn = nn.SmoothL1Loss(beta=float(cfg.get("beta", 0.5)))
        X_tn = torch.tensor(X_fit, dtype=torch.float32, device=device)
        y_tn = torch.tensor(y_fit, dtype=torch.float32, device=device)
        X_watch_tn = torch.tensor(X_watch, dtype=torch.float32, device=device) if X_watch is not None else None
        y_watch_np = np.asarray(y_watch, dtype=np.float32) if y_watch is not None else None
        batch_size = int(max(1, cfg.get("batch_size", 64)))
        best_state = None
        best_epoch = 0
        best_mae = float("inf")
        stale = 0
        epochs = int(max(1, max_epochs))
        for epoch in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(len(X_tn), device=device)
            for start in range(0, len(X_tn), batch_size):
                batch = perm[start : start + batch_size]
                pred = model(X_tn[batch])
                loss = loss_fn(pred, y_tn[batch])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if X_watch_tn is None or y_watch_np is None:
                continue
            model.eval()
            with torch.no_grad():
                pred_watch = model(X_watch_tn).detach().cpu().numpy()
            mae = float(np.mean(np.abs(pred_watch - y_watch_np)))
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        if best_epoch <= 0:
            best_epoch = epochs
        return model, best_epoch

    xs, ys = _fit_scalers(Z_train, y_train)
    Z_tr_s, y_tr_s = _prepare(xs, ys, Z_train, y_train)
    Z_va_s, y_va_s = _prepare(xs, ys, Z_val, y_val)
    Z_te_s, _ = _prepare(xs, ys, Z_test)
    trained = _train_once(
        Z_tr_s,
        y_tr_s,
        Z_va_s,
        y_va_s,
        max_epochs=int(cfg.get("epochs", 300)),
        patience=int(cfg.get("patience", 30)),
        seed_offset=0,
    )
    if trained is None:
        return None
    model, best_epoch = trained
    model.eval()
    with torch.no_grad():
        val_s = model(torch.tensor(Z_va_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
        if not refit_on_train_val:
            test_s = model(torch.tensor(Z_te_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
            val_raw = ys.inverse_transform(val_s.reshape(-1, 1)).ravel().astype(np.float32)
            test_raw = ys.inverse_transform(test_s.reshape(-1, 1)).ravel().astype(np.float32)
            return val_raw, test_raw, {"type": "hyper_emb_adapter", "best_epoch": best_epoch, "refit": False}

    Z_tv = np.concatenate([Z_train, Z_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)
    xs_final, ys_final = _fit_scalers(Z_tv, y_tv)
    Z_tv_s, y_tv_s = _prepare(xs_final, ys_final, Z_tv, y_tv)
    Z_te_final_s, _ = _prepare(xs_final, ys_final, Z_test)
    final_trained = _train_once(
        Z_tv_s,
        y_tv_s,
        None,
        None,
        max_epochs=best_epoch,
        patience=best_epoch + 1,
        seed_offset=1,
    )
    if final_trained is None:
        return None
    final_model, _ = final_trained
    final_model.eval()
    with torch.no_grad():
        test_s = final_model(torch.tensor(Z_te_final_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
    val_raw = ys.inverse_transform(val_s.reshape(-1, 1)).ravel().astype(np.float32)
    test_raw = ys_final.inverse_transform(test_s.reshape(-1, 1)).ravel().astype(np.float32)
    return val_raw, test_raw, {"type": "hyper_emb_adapter", "best_epoch": best_epoch, "refit": True}

from __future__ import annotations

from dataclasses import dataclass
import re
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
    candidate_diagnostics: list[dict[str, Any]] | None = None


def _build_candidate_context(
    *,
    model_cfg: dict[str, Any],
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
    source_context_tokens: list[tuple[str, ...]] | None,
    target_mol_keys: np.ndarray | None,
    target_context_tokens: list[tuple[str, ...]] | None,
    target_dataset_id: str | None,
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
    y_test_used = y_target[split.test_idx]
    y_test_sec = (
        np.asarray(y_target_sec, dtype=np.float32)[split.test_idx]
        if y_target_sec is not None
        else _inverse_target(y_test_used, target_transform, target_inv_scale, target_t0_sec)
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
    target_tokens_ordered = None
    if target_context_tokens is not None:
        ordered_idx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
        tokens_arr = list(target_context_tokens)
        target_tokens_ordered = [tokens_arr[int(i)] if int(i) < len(tokens_arr) else tuple() for i in ordered_idx]
    return CandidateBuildContext(
        model_cfg=model_cfg,
        X_src=X_src,
        X_src_mol=X_src_mol,
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
        y_test_sec=y_test_sec,
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
        source_context_tokens=source_context_tokens,
        target_mol_keys=target_keys,
        target_context_tokens=target_tokens_ordered,
        target_cp_reference=np.asarray(X_target_cp[0], dtype=np.float32) if len(X_target_cp) else None,
        target_dataset_id=str(target_dataset_id).zfill(4) if target_dataset_id is not None else None,
        X_val_cp=X_target_cp[split.val_idx],
        X_test_cp=X_target_cp[split.test_idx],
    )

def _priority_rank(candidate_name: str, patterns: list[re.Pattern[str]]) -> int:
    for index, pattern in enumerate(patterns):
        if pattern.search(candidate_name):
            return index
    return len(patterns)


def _priority_patterns(raw: Any) -> list[re.Pattern[str]]:
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
            patterns.append(re.compile(text))
    return patterns


def _required_patterns(raw: Any) -> list[re.Pattern[str]]:
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    patterns: list[re.Pattern[str]] = []
    for value in values:
        text = str(value or "").strip()
        if text:
            patterns.append(re.compile(text, flags=re.IGNORECASE))
    return patterns


def _tokens_match_all(tokens: list[tuple[str, ...]], patterns: list[re.Pattern[str]]) -> bool:
    if not patterns:
        return True
    texts = [" ".join(str(tok) for tok in row_tokens) for row_tokens in tokens]
    for pattern in patterns:
        if not any(pattern.search(text) for text in texts if text):
            return False
    return True


def _split_context_tokens(ctx: CandidateBuildContext) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]], list[tuple[str, ...]]]:
    total = len(ctx.y_train_sec) + len(ctx.y_val_sec) + len(ctx.X_test)
    tokens = list(ctx.target_context_tokens or [])
    if len(tokens) != total:
        tokens = [tuple()] * total
    n_tr = len(ctx.y_train_sec)
    n_va = len(ctx.y_val_sec)
    return tokens[:n_tr], tokens[n_tr : n_tr + n_va], tokens[n_tr + n_va :]


def _parse_fusion_target_quantile_rules(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    rules: list[dict[str, Any]] = []
    for item in values:
        if isinstance(item, dict):
            pattern_text = str(item.get("pattern", "") or "").strip()
            if not pattern_text:
                continue
            try:
                quantile = float(item.get("quantile"))
            except (TypeError, ValueError):
                continue
            rules.append(
                {
                    "pattern_text": pattern_text,
                    "pattern": re.compile(pattern_text, flags=re.IGNORECASE),
                    "quantile": float(np.clip(quantile, 0.0, 1.0)),
                    "blend": item.get("blend", None),
                    "target_requires": _required_patterns(
                        item.get("target_requires", item.get("requires", item.get("target_require", [])))
                    ),
                    "min_ref_iqr": item.get("min_ref_iqr", None),
                    "max_ref_iqr": item.get("max_ref_iqr", None),
                }
            )
            continue
        text = str(item or "").strip()
        if "=>" not in text:
            continue
        pattern_text, quantile_text = text.rsplit("=>", 1)
        pattern_text = pattern_text.strip()
        if not pattern_text:
            continue
        try:
            quantile = float(quantile_text.strip())
        except ValueError:
            continue
        rules.append(
            {
                "pattern_text": pattern_text,
                "pattern": re.compile(pattern_text, flags=re.IGNORECASE),
                "quantile": float(np.clip(quantile, 0.0, 1.0)),
                "blend": None,
                "target_requires": [],
                "min_ref_iqr": None,
                "max_ref_iqr": None,
            }
        )
    return rules


def _apply_fusion_target_quantile_rules(
    pred: np.ndarray,
    row_tokens: list[tuple[str, ...]],
    y_ref_sec: np.ndarray,
    all_target_tokens: list[tuple[str, ...]],
    rules: list[dict[str, Any]],
    blend: float,
    diagnostics: list[dict[str, Any]] | None = None,
    max_abs_shift: float = 0.0,
    max_shift_ref_iqr_mult: float = 0.0,
) -> np.ndarray:
    out = np.asarray(pred, dtype=np.float32).copy()
    if not rules or len(row_tokens) != len(out):
        return out
    ref = np.asarray(y_ref_sec, dtype=np.float32).reshape(-1)
    finite_ref = ref[np.isfinite(ref)]
    if finite_ref.size == 0:
        return out
    ref_iqr = float(np.nanquantile(finite_ref, 0.75) - np.nanquantile(finite_ref, 0.25))
    shift_caps: list[float] = []
    if float(max_abs_shift) > 0.0:
        shift_caps.append(float(max_abs_shift))
    if float(max_shift_ref_iqr_mult) > 0.0 and np.isfinite(ref_iqr) and ref_iqr > 0.0:
        shift_caps.append(float(max_shift_ref_iqr_mult) * ref_iqr)
    shift_cap = min(shift_caps) if shift_caps else 0.0
    target_ok_cache: dict[int, bool] = {}
    quantile_cache: dict[float, float] = {}
    w = float(np.clip(blend, 0.0, 1.0))
    for row_idx, tokens in enumerate(row_tokens):
        text = " ".join(str(tok) for tok in tokens)
        if not text:
            continue
        for rule_idx, rule in enumerate(rules):
            if rule_idx not in target_ok_cache:
                target_ok_cache[rule_idx] = _tokens_match_all(
                    all_target_tokens,
                    list(rule.get("target_requires", [])),
                )
            if not target_ok_cache[rule_idx]:
                continue
            min_iqr = rule.get("min_ref_iqr", None)
            if min_iqr is not None and ref_iqr < float(min_iqr):
                continue
            max_iqr = rule.get("max_ref_iqr", None)
            if max_iqr is not None and ref_iqr > float(max_iqr):
                continue
            if not rule["pattern"].search(text):
                continue
            q = float(rule["quantile"])
            if q not in quantile_cache:
                quantile_cache[q] = float(np.nanquantile(finite_ref, q))
            rule_blend = rule.get("blend", None)
            try:
                row_w = float(np.clip(w if rule_blend is None else float(rule_blend), 0.0, 1.0))
            except (TypeError, ValueError):
                row_w = w
            old_value = float(out[row_idx])
            uncapped_value = float(((1.0 - row_w) * old_value + row_w * quantile_cache[q]))
            new_value = uncapped_value
            capped = False
            if shift_cap > 0.0:
                shift = float(np.clip(uncapped_value - old_value, -shift_cap, shift_cap))
                new_value = old_value + shift
                capped = bool(abs(new_value - uncapped_value) > 1e-6)
            out[row_idx] = np.float32(new_value)
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "pattern": str(rule.get("pattern_text", "")),
                        "quantile": q,
                        "blend": row_w,
                        "row_idx": int(row_idx),
                        "old": old_value,
                        "new": new_value,
                        "uncapped_new": uncapped_value,
                        "shift": new_value - old_value,
                        "abs_shift": abs(new_value - old_value),
                        "shift_cap": shift_cap,
                        "capped": capped,
                    }
                )
            break
    return out


def _summarize_quantile_rule_events(prefix: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    key_prefix = f"fusion_quantile_{_safe_diagnostic_key(prefix)}"
    if not events:
        return {
            f"{key_prefix}_adjusted_count": 0,
            f"{key_prefix}_rule_count": 0,
            f"{key_prefix}_mean_abs_shift": 0.0,
            f"{key_prefix}_max_abs_shift": 0.0,
            f"{key_prefix}_capped_count": 0,
            f"{key_prefix}_rules": "",
        }
    grouped: dict[tuple[str, float, float], dict[str, float | int | str]] = {}
    abs_shifts: list[float] = []
    capped_count = 0
    for event in events:
        pattern = str(event.get("pattern", ""))
        quantile = float(event.get("quantile", 0.0))
        blend = float(event.get("blend", 0.0))
        key = (pattern, quantile, blend)
        cur = grouped.setdefault(
            key,
            {"pattern": pattern, "quantile": quantile, "blend": blend, "count": 0, "sum_abs": 0.0, "max_abs": 0.0},
        )
        shift = float(event.get("abs_shift", 0.0))
        cur["count"] = int(cur["count"]) + 1
        cur["sum_abs"] = float(cur["sum_abs"]) + shift
        cur["max_abs"] = max(float(cur["max_abs"]), shift)
        abs_shifts.append(shift)
        if bool(event.get("capped", False)):
            capped_count += 1
    parts = []
    for item in sorted(grouped.values(), key=lambda value: (-int(value["count"]), str(value["pattern"]))):
        count = int(item["count"])
        mean_abs = float(item["sum_abs"]) / max(count, 1)
        parts.append(
            f"{item['pattern']}@q={float(item['quantile']):.3g},b={float(item['blend']):.3g},n={count},mean_abs={mean_abs:.4g},max_abs={float(item['max_abs']):.4g}"
        )
    return {
        f"{key_prefix}_adjusted_count": int(len(events)),
        f"{key_prefix}_rule_count": int(len(grouped)),
        f"{key_prefix}_mean_abs_shift": float(np.mean(abs_shifts)) if abs_shifts else 0.0,
        f"{key_prefix}_max_abs_shift": float(np.max(abs_shifts)) if abs_shifts else 0.0,
        f"{key_prefix}_capped_count": int(capped_count),
        f"{key_prefix}_rules": "; ".join(parts),
    }


def _cfg_threshold(model_cfg: dict[str, Any], key: str) -> float | None:
    if key not in model_cfg or model_cfg.get(key) is None:
        return None
    try:
        value = float(model_cfg.get(key))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _priority_rank_for_candidate(
    candidate: Any,
    priority_patterns: list[re.Pattern[str]] | None,
    model_cfg: dict[str, Any],
) -> int:
    patterns = priority_patterns or []
    priority = _priority_rank(str(candidate.name), patterns)
    if priority >= len(patterns):
        return priority
    min_val_r2 = _cfg_threshold(model_cfg, "FUSION_PRIORITY_MIN_VAL_R2")
    if min_val_r2 is not None:
        val_r2 = float(candidate.val_metrics.get("r2", float("nan")))
        if not np.isfinite(val_r2) or val_r2 < min_val_r2:
            return len(patterns) + 1
    return priority


def _sort_key(
    candidate: Any,
    rank_mode: str,
    priority_patterns: list[re.Pattern[str]] | None = None,
    model_cfg: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    priority = _priority_rank_for_candidate(candidate, priority_patterns or [], model_cfg or {})
    mae = float(candidate.val_metrics.get("mae", float("inf")))
    if rank_mode == "mae_then_r2":
        return float(priority), mae, -float(candidate.val_metrics.get("r2", float("-inf")))
    return float(priority), mae, 0.0


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


def _selection_mask_from_y(y_val_sec: np.ndarray, trim_frac: float) -> np.ndarray:
    y = np.asarray(y_val_sec, dtype=np.float64).reshape(-1)
    mask = np.ones(len(y), dtype=bool)
    if not (0.0 < float(trim_frac) < 0.5) or len(y) < 5:
        return mask
    keep = np.isfinite(y)
    if int(np.sum(keep)) < 5:
        return mask
    lo = float(np.nanquantile(y[keep], float(trim_frac)))
    hi = float(np.nanquantile(y[keep], 1.0 - float(trim_frac)))
    trimmed = keep & (y >= lo) & (y <= hi)
    if int(np.sum(trimmed)) >= max(3, len(y) // 2):
        return trimmed
    return mask


def _filter_low_iqr_candidates(candidates: list[Any], model_cfg: dict[str, Any], y_train_sec: np.ndarray) -> list[Any]:
    max_iqr = float(model_cfg.get("FUSION_LOW_IQR_MAX_REF_IQR", 0.0) or 0.0)
    deny_patterns = _priority_patterns(model_cfg.get("FUSION_LOW_IQR_DENY_PATTERNS"))
    if max_iqr <= 0.0 or not deny_patterns:
        return candidates
    y = np.asarray(y_train_sec, dtype=np.float64).reshape(-1)
    finite = y[np.isfinite(y)]
    if finite.size < 5:
        return candidates
    ref_iqr = float(np.nanquantile(finite, 0.75) - np.nanquantile(finite, 0.25))
    if not np.isfinite(ref_iqr) or ref_iqr > max_iqr:
        return candidates
    filtered = [
        candidate
        for candidate in candidates
        if not any(pattern.search(str(candidate.name)) for pattern in deny_patterns)
    ]
    return filtered or candidates


def _filter_close_linear_prior_calibration_candidates(
    candidates: list[Any],
    model_cfg: dict[str, Any],
    *,
    target_rows: int | None = None,
) -> list[Any]:
    max_delta = float(model_cfg.get("FUSION_PREFER_ISOTONIC_PRIOR_CAL_MAX_MAE_DELTA", 0.0) or 0.0)
    max_ratio = float(model_cfg.get("FUSION_PREFER_ISOTONIC_PRIOR_CAL_MAX_MAE_RATIO", 0.0) or 0.0)
    if max_delta <= 0.0 and max_ratio <= 0.0:
        return candidates
    min_target_rows = int(model_cfg.get("FUSION_PREFER_ISOTONIC_PRIOR_CAL_MIN_TARGET_ROWS", 0) or 0)
    if min_target_rows > 0 and target_rows is not None and int(target_rows) < min_target_rows:
        return candidates

    linear = [
        candidate
        for candidate in candidates
        if str(candidate.name).startswith("HYPER_PRIOR_CAL_LINEAR")
    ]
    isotonic = [
        candidate
        for candidate in candidates
        if str(candidate.name).startswith("HYPER_PRIOR_CAL_ISOTONIC")
    ]
    if not linear or not isotonic:
        return candidates

    best_iso_mae = min(float(candidate.val_metrics.get("mae", float("inf"))) for candidate in isotonic)
    if not np.isfinite(best_iso_mae):
        return candidates

    remove: set[int] = set()
    for candidate in linear:
        linear_mae = float(candidate.val_metrics.get("mae", float("inf")))
        if not np.isfinite(linear_mae):
            continue
        close_by_delta = max_delta > 0.0 and best_iso_mae <= linear_mae + max_delta
        close_by_ratio = max_ratio > 0.0 and best_iso_mae <= max(linear_mae, 1e-8) * max_ratio
        if close_by_delta or close_by_ratio:
            remove.add(id(candidate))

    filtered = [candidate for candidate in candidates if id(candidate) not in remove]
    return filtered or candidates


def _prior_calibration_diagnostics(candidate: Any) -> dict[str, Any]:
    model = getattr(candidate, "model", None)
    if not isinstance(model, dict):
        return {}
    diagnostics = model.get("prior_calibration", {})
    if isinstance(diagnostics, dict):
        return diagnostics
    nested = model.get("diagnostics", {})
    if isinstance(nested, dict) and isinstance(nested.get("prior_calibration"), dict):
        return nested["prior_calibration"]
    return {}


def _safe_diagnostic_key(raw: Any) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(raw).strip().lower()).strip("_")
    return text or "unknown"


def _append_similarity_diagnostics(row: dict[str, Any], candidate: Any) -> None:
    model = getattr(candidate, "model", None)
    if not isinstance(model, dict):
        return
    diagnostics = model.get("similarity_diagnostics", {})
    if not isinstance(diagnostics, dict):
        return
    try:
        row["sim_n_test"] = int(diagnostics.get("n_test", 0))
    except (TypeError, ValueError):
        pass
    spaces = diagnostics.get("spaces", {})
    if not isinstance(spaces, dict):
        return
    for space_name, refs in spaces.items():
        if not isinstance(refs, dict):
            continue
        space_key = _safe_diagnostic_key(space_name)
        for ref_name, summary in refs.items():
            if not isinstance(summary, dict):
                continue
            ref_key = _safe_diagnostic_key(ref_name)
            for metric_name, value in summary.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(numeric):
                    continue
                row[f"sim_{space_key}_{ref_key}_{_safe_diagnostic_key(metric_name)}"] = numeric


def _diag_float(diag: dict[str, Any], section: str, key: str) -> float:
    section_diag = diag.get(section, {})
    if not isinstance(section_diag, dict):
        return float("nan")
    try:
        return float(section_diag.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _filter_risky_linear_prior_calibration_candidates(
    candidates: list[Any],
    model_cfg: dict[str, Any],
) -> list[Any]:
    max_abs_slope = _cfg_threshold(model_cfg, "FUSION_PRIOR_CAL_LINEAR_MAX_ABS_SLOPE")
    min_abs_slope = _cfg_threshold(model_cfg, "FUSION_PRIOR_CAL_LINEAR_MIN_ABS_SLOPE")
    max_test_extrap_frac = _cfg_threshold(model_cfg, "FUSION_PRIOR_CAL_LINEAR_MAX_TEST_EXTRAP_FRAC")
    max_test_extrap_ratio = _cfg_threshold(model_cfg, "FUSION_PRIOR_CAL_LINEAR_MAX_TEST_EXTRAP_RATIO")
    max_test_vs_val_extrap_delta = _cfg_threshold(model_cfg, "FUSION_PRIOR_CAL_LINEAR_MAX_TEST_VS_VAL_EXTRAP_FRAC_DELTA")
    min_fit_range = _cfg_threshold(model_cfg, "FUSION_PRIOR_CAL_LINEAR_MIN_FIT_RANGE")
    max_test_range_ratio = _cfg_threshold(model_cfg, "FUSION_PRIOR_CAL_LINEAR_MAX_TEST_RANGE_RATIO")
    thresholds = [
        max_abs_slope,
        min_abs_slope,
        max_test_extrap_frac,
        max_test_extrap_ratio,
        max_test_vs_val_extrap_delta,
        min_fit_range,
        max_test_range_ratio,
    ]
    if all(value is None for value in thresholds):
        return candidates

    if bool(model_cfg.get("FUSION_PRIOR_CAL_LINEAR_FILTER_REQUIRE_ISOTONIC", False)):
        if not any(str(candidate.name).startswith("HYPER_PRIOR_CAL_ISOTONIC") for candidate in candidates):
            return candidates

    remove: set[int] = set()
    for candidate in candidates:
        if not str(candidate.name).startswith("HYPER_PRIOR_CAL_LINEAR"):
            continue
        diag = _prior_calibration_diagnostics(candidate)
        if not diag:
            continue
        slope_abs = _diag_float(diag, "test", "slope_abs_max")
        test_fit_range = _diag_float(diag, "test", "fit_pred_range_min")
        test_extrap_frac = _diag_float(diag, "test", "eval_extrapolation_frac_max")
        val_extrap_frac = _diag_float(diag, "val", "eval_extrapolation_frac_max")
        test_extrap_ratio = _diag_float(diag, "test", "eval_extrapolation_ratio_max")
        test_range_ratio = _diag_float(diag, "test", "eval_to_fit_range_ratio_max")

        risky = False
        if max_abs_slope is not None and np.isfinite(slope_abs) and slope_abs > max_abs_slope:
            risky = True
        if min_abs_slope is not None and np.isfinite(slope_abs) and slope_abs < min_abs_slope:
            risky = True
        if max_test_extrap_frac is not None and np.isfinite(test_extrap_frac) and test_extrap_frac > max_test_extrap_frac:
            risky = True
        if max_test_extrap_ratio is not None and np.isfinite(test_extrap_ratio) and test_extrap_ratio > max_test_extrap_ratio:
            risky = True
        if (
            max_test_vs_val_extrap_delta is not None
            and np.isfinite(test_extrap_frac)
            and np.isfinite(val_extrap_frac)
            and (test_extrap_frac - val_extrap_frac) > max_test_vs_val_extrap_delta
        ):
            risky = True
        if min_fit_range is not None and np.isfinite(test_fit_range) and test_fit_range < min_fit_range:
            risky = True
        if max_test_range_ratio is not None and np.isfinite(test_range_ratio) and test_range_ratio > max_test_range_ratio:
            risky = True
        if risky:
            remove.add(id(candidate))

    filtered = [candidate for candidate in candidates if id(candidate) not in remove]
    return filtered or candidates


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
    source_context_tokens: list[tuple[str, ...]] | None = None,
    target_mol_keys: np.ndarray | None = None,
    target_context_tokens: list[tuple[str, ...]] | None = None,
    target_dataset_id: str | None = None,
) -> EnsembleOutput:
    ctx = _build_candidate_context(
        model_cfg=model_cfg,
        X_src=X_src,
        X_src_mol=X_src_mol,
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
        source_context_tokens=source_context_tokens,
        target_mol_keys=target_mol_keys,
        target_context_tokens=target_context_tokens,
        target_dataset_id=target_dataset_id,
    )
    priority_patterns = _priority_patterns(model_cfg.get("FUSION_PRIORITY_PATTERNS"))
    candidates = sorted(
        build_candidates(ctx),
        key=lambda candidate: _sort_key(
            candidate,
            str(model_cfg.get("FUSION_RANK", "mae")).strip().lower(),
            priority_patterns,
            model_cfg,
        ),
    )

    if bool(model_cfg.get("CALIBRATE", True)):
        calibrate_candidates(candidates, ctx.y_val_sec)

    preserve_candidate_selection_metrics = bool(model_cfg.get("PRESERVE_CANDIDATE_SELECTION_METRICS", False)) or bool(
        model_cfg.get("CALIBRATED_MOL_INTERNAL_CV_METRICS", False)
    )
    clip_mult = float(model_cfg.get("CLIP_MULT", 1.5))
    y_clip_max = float(np.nanmax(ctx.y_train_sec)) * clip_mult if len(ctx.y_train_sec) else float("inf")
    if np.isfinite(y_clip_max) and y_clip_max > 0:
        for candidate in candidates:
            candidate.val_pred = np.clip(candidate.val_pred, 0.0, y_clip_max)
            candidate.test_pred = np.clip(candidate.test_pred, 0.0, y_clip_max)
            if not preserve_candidate_selection_metrics:
                candidate.val_metrics = compute_metrics(ctx.y_val_sec, candidate.val_pred)
    upper_q = float(model_cfg.get("PRED_CLIP_UPPER_TRAIN_QUANTILE", 0.0) or 0.0)
    lower_q = float(model_cfg.get("PRED_CLIP_LOWER_TRAIN_QUANTILE", 0.0) or 0.0)
    if len(ctx.y_train_sec) and 0.0 < upper_q <= 1.0:
        y_clip_q_max = float(np.nanquantile(ctx.y_train_sec, upper_q))
        if np.isfinite(y_clip_q_max) and y_clip_q_max > 0.0:
            for candidate in candidates:
                candidate.val_pred = np.clip(candidate.val_pred, 0.0, y_clip_q_max)
                candidate.test_pred = np.clip(candidate.test_pred, 0.0, y_clip_q_max)
                if not preserve_candidate_selection_metrics:
                    candidate.val_metrics = compute_metrics(ctx.y_val_sec, candidate.val_pred)
    if len(ctx.y_train_sec) and 0.0 < lower_q < 1.0:
        y_clip_q_min = float(np.nanquantile(ctx.y_train_sec, lower_q))
        if np.isfinite(y_clip_q_min) and y_clip_q_min > 0.0:
            for candidate in candidates:
                candidate.val_pred = np.maximum(candidate.val_pred, y_clip_q_min)
                candidate.test_pred = np.maximum(candidate.test_pred, y_clip_q_min)
                if not preserve_candidate_selection_metrics:
                    candidate.val_metrics = compute_metrics(ctx.y_val_sec, candidate.val_pred)
    trim_frac = float(model_cfg.get("FUSION_VALIDATION_TRIM_FRAC", 0.0) or 0.0)
    selection_mask = _selection_mask_from_y(ctx.y_val_sec, trim_frac)
    selection_y = np.asarray(ctx.y_val_sec, dtype=np.float32)[selection_mask]
    if int(np.sum(selection_mask)) != len(ctx.y_val_sec):
        for candidate in candidates:
            candidate.val_metrics = compute_metrics(selection_y, np.asarray(candidate.val_pred, dtype=np.float32)[selection_mask])
    rank_mode = str(model_cfg.get("FUSION_RANK", "mae")).strip().lower()
    candidates = sorted(candidates, key=lambda candidate: _sort_key(candidate, rank_mode, priority_patterns, model_cfg))
    candidates = [candidate for candidate in candidates if _is_valid_candidate(candidate, expected_val_len=len(ctx.y_val_sec))]
    if not candidates:
        raise ValueError("No valid ensemble candidates were produced.")
    candidates = _filter_low_iqr_candidates(candidates, model_cfg, ctx.y_train_sec)
    candidates = _filter_close_linear_prior_calibration_candidates(
        candidates,
        model_cfg,
        target_rows=len(ctx.X_train) + len(ctx.X_val) + len(ctx.X_test),
    )
    candidates = _filter_risky_linear_prior_calibration_candidates(candidates, model_cfg)
    if bool(model_cfg.get("FUSION_EXCLUSIVE_PRIORITY", False)) and priority_patterns:
        best_priority = min(_priority_rank_for_candidate(candidate, priority_patterns, model_cfg) for candidate in candidates)
        if best_priority < len(priority_patterns):
            candidates = [
                candidate
                for candidate in candidates
                if _priority_rank_for_candidate(candidate, priority_patterns, model_cfg) == best_priority
            ]
    max_val_mae_ratio = float(model_cfg.get("FUSION_MAX_VAL_MAE_RATIO", 0.0) or 0.0)
    if max_val_mae_ratio > 0.0:
        best_mae = min(float(candidate.val_metrics.get("mae", float("inf"))) for candidate in candidates)
        if np.isfinite(best_mae):
            threshold = max(best_mae, 1e-8) * max_val_mae_ratio
            candidates = [
                candidate
                for candidate in candidates
                if float(candidate.val_metrics.get("mae", float("inf"))) <= threshold
            ]

    top = candidates[: int(model_cfg.get("FUSION_TOP_K", 3))]
    if not top:
        raise ValueError("No ensemble candidates available after ranking.")
    val_mat_full = np.column_stack([candidate.val_pred for candidate in top])
    val_mat = val_mat_full[selection_mask]
    test_mat = np.column_stack([candidate.test_pred for candidate in top])
    weights = _optimize_weights(
        selection_y,
        val_mat,
        objective=str(model_cfg.get("FUSION_OBJECTIVE", "mae")),
        r2_weight=float(model_cfg.get("FUSION_R2_WEIGHT", 0.25)),
        l2_reg=float(model_cfg.get("FUSION_L2_REG", 0.0)),
        weight_floor=float(model_cfg.get("FUSION_WEIGHT_FLOOR", 0.0)),
    )
    weight_by_rank = {rank: float(weight) for rank, weight in enumerate(weights)}
    include_test_diagnostics = bool(model_cfg.get("DIAGNOSTIC_INCLUDE_TEST_METRICS", False))
    candidate_diagnostics = []
    for rank, candidate in enumerate(candidates):
        weight = float(weight_by_rank.get(rank, 0.0))
        row = {
            "rank": int(rank + 1),
            "name": str(candidate.name),
            "val_mae": float(candidate.val_metrics.get("mae", float("nan"))),
            "val_r2": float(candidate.val_metrics.get("r2", float("nan"))),
            "selected": bool(abs(weight) > 1e-12),
            "weight": weight,
        }
        prior_diag = _prior_calibration_diagnostics(candidate)
        if prior_diag:
            row.update(
                {
                    "prior_cal_mode": str(prior_diag.get("mode", "")),
                    "prior_cal_test_slope_abs_max": _diag_float(prior_diag, "test", "slope_abs_max"),
                    "prior_cal_test_extrap_frac_max": _diag_float(prior_diag, "test", "eval_extrapolation_frac_max"),
                    "prior_cal_test_extrap_ratio_max": _diag_float(prior_diag, "test", "eval_extrapolation_ratio_max"),
                    "prior_cal_test_fit_pred_range_min": _diag_float(prior_diag, "test", "fit_pred_range_min"),
                    "prior_cal_test_range_ratio_max": _diag_float(prior_diag, "test", "eval_to_fit_range_ratio_max"),
                }
            )
        _append_similarity_diagnostics(row, candidate)
        if include_test_diagnostics and ctx.y_test_sec is not None:
            test_metrics = compute_metrics(ctx.y_test_sec, candidate.test_pred)
            row.update(
                {
                    "test_mae": float(test_metrics.get("mae", float("nan"))),
                    "test_r2": float(test_metrics.get("r2", float("nan"))),
                }
            )
        candidate_diagnostics.append(row)
    pred_val = val_mat_full @ weights
    pred_test = test_mat @ weights
    pre_quantile_row = {
        "rank": int(-1),
        "name": "FUSION_PRE_QUANTILE",
        "val_mae": float(compute_metrics(ctx.y_val_sec, pred_val).get("mae", float("nan"))),
        "val_r2": float(compute_metrics(ctx.y_val_sec, pred_val).get("r2", float("nan"))),
        "selected": True,
        "weight": 1.0,
    }
    if include_test_diagnostics and ctx.y_test_sec is not None:
        test_metrics = compute_metrics(ctx.y_test_sec, pred_test)
        pre_quantile_row.update(
            {
                "test_mae": float(test_metrics.get("mae", float("nan"))),
                "test_r2": float(test_metrics.get("r2", float("nan"))),
            }
        )
    candidate_diagnostics.append(pre_quantile_row)
    fusion_quantile_rules = _parse_fusion_target_quantile_rules(model_cfg.get("FUSION_TARGET_QUANTILE_RULES"))
    if fusion_quantile_rules:
        train_tokens, val_tokens, test_tokens = _split_context_tokens(ctx)
        all_target_tokens = train_tokens + val_tokens + test_tokens
        blend = float(model_cfg.get("FUSION_TARGET_QUANTILE_BLEND", 1.0))
        max_shift_sec = float(model_cfg.get("FUSION_TARGET_QUANTILE_MAX_SHIFT_SEC", 0.0) or 0.0)
        max_shift_iqr_mult = float(model_cfg.get("FUSION_TARGET_QUANTILE_MAX_SHIFT_REF_IQR_MULT", 0.0) or 0.0)
        val_quantile_events: list[dict[str, Any]] = []
        test_quantile_events: list[dict[str, Any]] = []
        pred_val = _apply_fusion_target_quantile_rules(
            pred_val,
            val_tokens,
            ctx.y_train_sec,
            all_target_tokens,
            fusion_quantile_rules,
            blend,
            diagnostics=val_quantile_events,
            max_abs_shift=max_shift_sec,
            max_shift_ref_iqr_mult=max_shift_iqr_mult,
        )
        pred_test = _apply_fusion_target_quantile_rules(
            pred_test,
            test_tokens,
            np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0),
            all_target_tokens,
            fusion_quantile_rules,
            blend,
            diagnostics=test_quantile_events,
            max_abs_shift=max_shift_sec,
            max_shift_ref_iqr_mult=max_shift_iqr_mult,
        )
        row = {
            "rank": int(0),
            "name": "FUSION_TARGET_QUANTILE_RULES",
            "val_mae": float(compute_metrics(ctx.y_val_sec, pred_val).get("mae", float("nan"))),
            "val_r2": float(compute_metrics(ctx.y_val_sec, pred_val).get("r2", float("nan"))),
            "selected": True,
            "weight": 1.0,
        }
        row.update(_summarize_quantile_rule_events("val", val_quantile_events))
        row.update(_summarize_quantile_rule_events("test", test_quantile_events))
        if include_test_diagnostics and ctx.y_test_sec is not None:
            test_metrics = compute_metrics(ctx.y_test_sec, pred_test)
            row.update(
                {
                    "test_mae": float(test_metrics.get("mae", float("nan"))),
                    "test_r2": float(test_metrics.get("r2", float("nan"))),
                }
            )
        candidate_diagnostics.append(row)
    return EnsembleOutput(
        pred_test=pred_test,
        pred_val=pred_val,
        top_models=[candidate.name for candidate in top],
        weights=[float(weight) for weight in weights],
        feature_group_importance=_feature_group_importance(top[0].model, group_sizes),
        candidate_diagnostics=candidate_diagnostics,
    )

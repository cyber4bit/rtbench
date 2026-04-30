from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

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


def build_exact_mol_lookup_candidates(ctx: CandidateBuildContext) -> list[CandidateOutput]:
    if not bool(ctx.model_cfg.get("ENABLE_EXACT_MOL_LOOKUP", False)):
        return []
    if ctx.source_mol_keys is None or ctx.target_mol_keys is None:
        return []
    n_train = len(ctx.y_train_sec)
    n_val = len(ctx.y_val_sec)
    if len(ctx.target_mol_keys) < n_train + n_val:
        return []
    tr_keys = np.asarray(ctx.target_mol_keys[:n_train], dtype=object)
    va_keys = np.asarray(ctx.target_mol_keys[n_train : n_train + n_val], dtype=object)
    te_keys = np.asarray(ctx.target_mol_keys[n_train + n_val :], dtype=object)
    if len(te_keys) == 0 or n_train == 0:
        return []

    source_keys = np.asarray(ctx.source_mol_keys, dtype=object)
    source_y = np.asarray(ctx.y_src_sec, dtype=np.float32)
    source_keep = np.ones(len(source_keys), dtype=bool)

    def id_set(raw: Any) -> set[str]:
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

    use_all_lookup_sources = bool(
        ctx.model_cfg.get("LOOKUP_USE_ALL_SOURCE_IDS", False)
        or ctx.model_cfg.get("EXACT_MOL_USE_ALL_SOURCE_IDS", False)
    )
    allowed_ids = set() if use_all_lookup_sources else id_set(ctx.model_cfg.get("EXACT_MOL_ALLOWED_SOURCE_IDS"))
    excluded_ids = id_set(ctx.model_cfg.get("EXACT_MOL_EXCLUDED_SOURCE_IDS"))
    if ctx.source_row_dataset_ids is not None and (allowed_ids or excluded_ids):
        source_ids = np.asarray(ctx.source_row_dataset_ids, dtype=object)
        if len(source_ids) == len(source_keep):
            norm_ids = np.asarray([str(value).zfill(4) for value in source_ids], dtype=object)
            if allowed_ids:
                source_keep &= np.asarray([value in allowed_ids for value in norm_ids], dtype=bool)
            if excluded_ids:
                source_keep &= np.asarray([value not in excluded_ids for value in norm_ids], dtype=bool)
    source_keys = source_keys[source_keep]
    source_y = source_y[source_keep]

    def build_lookup(extra_keys: np.ndarray, extra_y: np.ndarray) -> tuple[dict[str, float], float]:
        sums: dict[str, float] = {}
        weights: dict[str, float] = {}

        def add(keys: np.ndarray, values: np.ndarray, weight: float) -> None:
            for mol_key, value in zip(keys, values):
                key = clean_mol_key(mol_key)
                if not key or not np.isfinite(float(value)):
                    continue
                sums[key] = sums.get(key, 0.0) + float(weight) * float(value)
                weights[key] = weights.get(key, 0.0) + float(weight)

        add(source_keys, source_y, 1.0)
        add(extra_keys, np.asarray(extra_y, dtype=np.float32), float(ctx.model_cfg.get("EXACT_MOL_TARGET_WEIGHT", 2.0)))
        lookup = {key: sums[key] / weights[key] for key in sums if weights.get(key, 0.0) > 0.0}
        fallback = float(np.median(extra_y)) if len(extra_y) else 0.0
        return lookup, fallback

    val_lookup, val_fallback = build_lookup(tr_keys, ctx.y_train_sec)
    test_keys_extra = np.concatenate([tr_keys, va_keys], axis=0)
    test_y_extra = np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0)
    test_lookup, test_fallback = build_lookup(test_keys_extra, test_y_extra)

    def predict(keys: np.ndarray, lookup: dict[str, float], fallback: float) -> tuple[np.ndarray, float]:
        out = np.full(len(keys), fallback, dtype=np.float32)
        covered = 0
        for index, mol_key in enumerate(keys):
            key = clean_mol_key(mol_key)
            if key in lookup:
                out[index] = float(lookup[key])
                covered += 1
        coverage = covered / max(len(keys), 1)
        return out, coverage

    val_pred, val_coverage = predict(va_keys, val_lookup, val_fallback)
    min_val_coverage = float(ctx.model_cfg.get("EXACT_MOL_MIN_VAL_COVERAGE", 0.20))
    if val_coverage < min_val_coverage:
        return []
    test_pred, _ = predict(te_keys, test_lookup, test_fallback)
    return [
        CandidateOutput(
            name="EXACT_MOL_LOOKUP",
            val_pred=val_pred,
            test_pred=test_pred,
            val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
            model={"type": "exact_mol_lookup", "val_coverage": float(val_coverage)},
        )
    ]


def build_calibrated_mol_lookup_candidates(ctx: CandidateBuildContext) -> list[CandidateOutput]:
    if not bool(ctx.model_cfg.get("ENABLE_CALIBRATED_MOL_LOOKUP", False)):
        return []
    if ctx.source_mol_keys is None or ctx.target_mol_keys is None or ctx.source_row_dataset_ids is None:
        return []
    n_train = len(ctx.y_train_sec)
    n_val = len(ctx.y_val_sec)
    if len(ctx.target_mol_keys) < n_train + n_val:
        return []
    tr_keys = np.asarray(ctx.target_mol_keys[:n_train], dtype=object)
    va_keys = np.asarray(ctx.target_mol_keys[n_train : n_train + n_val], dtype=object)
    te_keys = np.asarray(ctx.target_mol_keys[n_train + n_val :], dtype=object)
    if len(te_keys) == 0 or n_train == 0:
        return []

    source_keys = np.asarray(ctx.source_mol_keys, dtype=object)
    source_y = np.asarray(ctx.y_src_sec, dtype=np.float32)
    source_ids = np.asarray(ctx.source_row_dataset_ids, dtype=object)
    if len(source_ids) != len(source_keys):
        return []
    use_all_lookup_sources = bool(
        ctx.model_cfg.get("LOOKUP_USE_ALL_SOURCE_IDS", False)
        or ctx.model_cfg.get("CALIBRATED_MOL_USE_ALL_SOURCE_IDS", False)
    )
    allowed_ids = set() if use_all_lookup_sources else _id_set(ctx.model_cfg.get("EXACT_MOL_ALLOWED_SOURCE_IDS"))
    excluded_ids = _id_set(ctx.model_cfg.get("EXACT_MOL_EXCLUDED_SOURCE_IDS"))
    keep = np.ones(len(source_keys), dtype=bool)
    norm_ids = np.asarray([str(value).zfill(4) for value in source_ids], dtype=object)
    if allowed_ids:
        keep &= np.asarray([value in allowed_ids for value in norm_ids], dtype=bool)
    if excluded_ids:
        keep &= np.asarray([value not in excluded_ids for value in norm_ids], dtype=bool)
    source_keys = source_keys[keep]
    source_y = source_y[keep]
    norm_ids = norm_ids[keep]
    if len(source_keys) == 0:
        return []

    source_maps: dict[str, dict[str, float]] = {}
    source_cp_by_id: dict[str, np.ndarray] = {}
    for dataset_id in sorted(set(str(x) for x in norm_ids.tolist())):
        mask = norm_ids == dataset_id
        source_maps[dataset_id] = _mean_by_mol_key(source_keys[mask], source_y[mask])
        if ctx.X_src_cp is not None and len(ctx.X_src_cp) == len(keep):
            cp_rows = np.asarray(ctx.X_src_cp, dtype=np.float32)[keep][mask]
            if len(cp_rows):
                source_cp_by_id[dataset_id] = np.asarray(cp_rows[0], dtype=np.float32)

    min_overlap = int(ctx.model_cfg.get("CALIBRATED_MOL_MIN_OVERLAP", 2))

    def fit_calibrations(target_lookup: dict[str, float]) -> dict[str, tuple[float, float]]:
        calibrations: dict[str, tuple[float, float]] = {}
        for dataset_id, source_lookup in source_maps.items():
            overlap = [key for key in target_lookup if key in source_lookup]
            if len(overlap) < min_overlap:
                continue
            x = np.asarray([source_lookup[key] for key in overlap], dtype=np.float64)
            y = np.asarray([target_lookup[key] for key in overlap], dtype=np.float64)
            if float(np.std(x)) < 1e-8:
                continue
            design = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
            try:
                slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            if np.isfinite(slope) and np.isfinite(intercept):
                calibrations[dataset_id] = (float(intercept), float(slope))
        return calibrations

    def fit_weighted_calibrations(target_lookup: dict[str, float]) -> dict[str, tuple[float, float, float]]:
        calibrations: dict[str, tuple[float, float, float]] = {}
        for dataset_id, source_lookup in source_maps.items():
            overlap = [key for key in target_lookup if key in source_lookup]
            if len(overlap) < min_overlap:
                continue
            x = np.asarray([source_lookup[key] for key in overlap], dtype=np.float64)
            y = np.asarray([target_lookup[key] for key in overlap], dtype=np.float64)
            if float(np.std(x)) < 1e-8:
                continue
            design = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
            try:
                slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            pred = slope * x + intercept
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            if np.isfinite(slope) and np.isfinite(intercept) and np.isfinite(rmse):
                calibrations[dataset_id] = (float(intercept), float(slope), max(rmse, 1.0))
        return calibrations

    def fit_spline_calibrations(target_lookup: dict[str, float]) -> dict[str, tuple[Any, float]]:
        calibrations: dict[str, tuple[Any, float]] = {}
        spline_min_overlap = int(ctx.model_cfg.get("SPLINE_CALIBRATED_MOL_MIN_OVERLAP", max(8, min_overlap)))
        max_knots = int(ctx.model_cfg.get("SPLINE_CALIBRATED_MOL_N_KNOTS", 5))
        degree = int(ctx.model_cfg.get("SPLINE_CALIBRATED_MOL_DEGREE", 3))
        alpha = float(ctx.model_cfg.get("SPLINE_CALIBRATED_MOL_ALPHA", 1.0))
        for dataset_id, source_lookup in source_maps.items():
            overlap = [key for key in target_lookup if key in source_lookup]
            if len(overlap) < spline_min_overlap:
                continue
            x = np.asarray([source_lookup[key] for key in overlap], dtype=np.float64)
            y = np.asarray([target_lookup[key] for key in overlap], dtype=np.float64)
            keep = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(keep)) < spline_min_overlap or len(np.unique(x[keep])) < 4:
                continue
            x_fit = x[keep].reshape(-1, 1)
            y_fit = y[keep]
            n_knots = max(2, min(max_knots, len(np.unique(x[keep])) - 1))
            try:
                model = make_pipeline(
                    SplineTransformer(n_knots=n_knots, degree=max(1, degree), include_bias=False),
                    Ridge(alpha=alpha),
                )
                model.fit(x_fit, y_fit)
                pred = np.asarray(model.predict(x_fit), dtype=np.float64)
            except Exception:
                continue
            rmse = float(np.sqrt(np.mean((pred - y_fit) ** 2)))
            if np.isfinite(rmse):
                calibrations[dataset_id] = (model, max(rmse, 1.0))
        return calibrations

    def predict(
        keys: np.ndarray,
        direct_lookup: dict[str, float],
        calibrations: dict[str, tuple[float, float]],
    ) -> tuple[np.ndarray, float]:
        fallback = float(np.median(list(direct_lookup.values()))) if direct_lookup else 0.0
        out = np.full(len(keys), fallback, dtype=np.float32)
        covered = 0
        for index, mol_key in enumerate(keys):
            key = clean_mol_key(mol_key)
            if key in direct_lookup:
                out[index] = float(direct_lookup[key])
                covered += 1
                continue
            preds: list[float] = []
            for dataset_id, source_lookup in source_maps.items():
                if key not in source_lookup or dataset_id not in calibrations:
                    continue
                intercept, slope = calibrations[dataset_id]
                pred = intercept + slope * float(source_lookup[key])
                if np.isfinite(pred):
                    preds.append(float(pred))
            if preds:
                out[index] = float(np.median(preds))
                covered += 1
        return out, covered / max(len(keys), 1)

    def predict_quantile(
        keys: np.ndarray,
        direct_lookup: dict[str, float],
        calibrations: dict[str, tuple[float, float]],
        quantile: float,
    ) -> tuple[np.ndarray, float]:
        fallback = float(np.median(list(direct_lookup.values()))) if direct_lookup else 0.0
        q = float(np.clip(float(quantile), 0.0, 1.0))
        out = np.full(len(keys), fallback, dtype=np.float32)
        covered = 0
        for index, mol_key in enumerate(keys):
            key = clean_mol_key(mol_key)
            if key in direct_lookup:
                out[index] = float(direct_lookup[key])
                covered += 1
                continue
            preds: list[float] = []
            for dataset_id, source_lookup in source_maps.items():
                if key not in source_lookup or dataset_id not in calibrations:
                    continue
                intercept, slope = calibrations[dataset_id]
                pred = intercept + slope * float(source_lookup[key])
                if np.isfinite(pred):
                    preds.append(float(pred))
            if preds:
                out[index] = float(np.quantile(np.asarray(preds, dtype=np.float64), q))
                covered += 1
        return out, covered / max(len(keys), 1)

    def predict_weighted(
        keys: np.ndarray,
        direct_lookup: dict[str, float],
        calibrations: dict[str, tuple[float, float, float]],
        *,
        use_direct_lookup: bool = True,
        cp_weighted: bool = False,
    ) -> tuple[np.ndarray, float]:
        fallback = float(np.median(list(direct_lookup.values()))) if direct_lookup else 0.0
        out = np.full(len(keys), fallback, dtype=np.float32)
        covered = 0
        cp_weight_power = float(ctx.model_cfg.get("CP_WEIGHTED_CALIBRATED_MOL_SIMILARITY_POWER", 2.0))
        cp_weight_floor = float(ctx.model_cfg.get("CP_WEIGHTED_CALIBRATED_MOL_SIMILARITY_FLOOR", 0.0))
        target_cp = np.asarray(ctx.target_cp_reference, dtype=np.float32) if ctx.target_cp_reference is not None else None
        for index, mol_key in enumerate(keys):
            key = clean_mol_key(mol_key)
            if use_direct_lookup and key in direct_lookup:
                out[index] = float(direct_lookup[key])
                covered += 1
                continue
            preds: list[float] = []
            weights: list[float] = []
            for dataset_id, source_lookup in source_maps.items():
                if key not in source_lookup or dataset_id not in calibrations:
                    continue
                intercept, slope, rmse = calibrations[dataset_id]
                pred = intercept + slope * float(source_lookup[key])
                if np.isfinite(pred):
                    preds.append(float(pred))
                    weight = float(1.0 / max(rmse, 1.0) ** 2)
                    if cp_weighted and target_cp is not None and dataset_id in source_cp_by_id:
                        sim = max(0.0, cosine(target_cp, source_cp_by_id[dataset_id]))
                        weight *= max(float(sim), cp_weight_floor) ** cp_weight_power
                    weights.append(weight)
            if preds:
                out[index] = float(np.average(np.asarray(preds, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))
                covered += 1
        return out, covered / max(len(keys), 1)

    def predict_spline(
        keys: np.ndarray,
        direct_lookup: dict[str, float],
        calibrations: dict[str, tuple[Any, float]],
    ) -> tuple[np.ndarray, float]:
        fallback = float(np.median(list(direct_lookup.values()))) if direct_lookup else 0.0
        out = np.full(len(keys), fallback, dtype=np.float32)
        covered = 0
        for index, mol_key in enumerate(keys):
            key = clean_mol_key(mol_key)
            if key in direct_lookup:
                out[index] = float(direct_lookup[key])
                covered += 1
                continue
            preds: list[float] = []
            weights: list[float] = []
            for dataset_id, source_lookup in source_maps.items():
                if key not in source_lookup or dataset_id not in calibrations:
                    continue
                model, rmse = calibrations[dataset_id]
                try:
                    pred = float(np.asarray(model.predict(np.array([[float(source_lookup[key])]], dtype=np.float64))).reshape(-1)[0])
                except Exception:
                    continue
                if np.isfinite(pred):
                    preds.append(pred)
                    weights.append(float(1.0 / max(rmse, 1.0) ** 2))
            if preds:
                out[index] = float(np.average(np.asarray(preds, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))
                covered += 1
        return out, covered / max(len(keys), 1)

    def source_projection_features(
        keys: np.ndarray,
        direct_lookup: dict[str, float],
        calibrations: dict[str, tuple[float, float, float]],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        fallback = float(np.median(list(direct_lookup.values()))) if direct_lookup else 0.0
        out = np.full(len(keys), fallback, dtype=np.float32)
        covered_mask = np.zeros(len(keys), dtype=np.float32)
        covered = 0
        for index, mol_key in enumerate(keys):
            key = clean_mol_key(mol_key)
            preds: list[float] = []
            weights: list[float] = []
            for dataset_id, source_lookup in source_maps.items():
                if key not in source_lookup or dataset_id not in calibrations:
                    continue
                intercept, slope, rmse = calibrations[dataset_id]
                pred = intercept + slope * float(source_lookup[key])
                if np.isfinite(pred):
                    preds.append(float(pred))
                    weights.append(float(1.0 / max(rmse, 1.0) ** 2))
            if preds:
                out[index] = float(np.average(np.asarray(preds, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))
                covered_mask[index] = 1.0
                covered += 1
        return out, covered_mask, covered / max(len(keys), 1)

    def source_vector_raw(
        keys: np.ndarray,
        calibrations: dict[str, tuple[float, float]],
        *,
        include_direct: bool = False,
        direct_lookup: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, float]:
        source_order = sorted(dataset_id for dataset_id in source_maps if dataset_id in calibrations)
        if not source_order:
            return np.empty((len(keys), 0), dtype=np.float32), 0.0
        raw = np.full((len(keys), len(source_order)), np.nan, dtype=np.float32)
        covered = 0
        for row_index, mol_key in enumerate(keys):
            key = clean_mol_key(mol_key)
            row_covered = False
            for col_index, dataset_id in enumerate(source_order):
                source_lookup = source_maps[dataset_id]
                if key not in source_lookup:
                    continue
                intercept, slope = calibrations[dataset_id]
                pred = intercept + slope * float(source_lookup[key])
                if np.isfinite(pred):
                    raw[row_index, col_index] = float(pred)
                    row_covered = True
            if row_covered:
                covered += 1
        features = [raw]
        finite = np.isfinite(raw)
        coverage = np.mean(finite, axis=1, where=np.ones_like(finite, dtype=bool)).reshape(-1, 1)
        with np.errstate(all="ignore"):
            median = np.nanmedian(raw, axis=1).reshape(-1, 1)
            mean = np.nanmean(raw, axis=1).reshape(-1, 1)
            spread = np.nanstd(raw, axis=1).reshape(-1, 1)
        features.extend([median, mean, spread, coverage.astype(np.float32)])
        if include_direct:
            direct = np.full((len(keys), 1), np.nan, dtype=np.float32)
            if direct_lookup:
                for row_index, mol_key in enumerate(keys):
                    key = clean_mol_key(mol_key)
                    if key in direct_lookup:
                        direct[row_index, 0] = float(direct_lookup[key])
            features.append(direct)
        return np.column_stack(features).astype(np.float32), covered / max(len(keys), 1)

    def fit_impute(train_raw: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
        train = np.asarray(train_raw, dtype=np.float32)
        if train.ndim != 2 or train.shape[1] == 0:
            return (train, *[np.asarray(other, dtype=np.float32) for other in others])
        mask_train = np.isfinite(train).astype(np.float32)
        with np.errstate(all="ignore"):
            fill = np.nanmedian(train, axis=0)
        global_fill = float(np.nanmedian(train)) if np.any(np.isfinite(train)) else 0.0
        fill = np.where(np.isfinite(fill), fill, global_fill).astype(np.float32)

        def transform(raw_matrix: np.ndarray) -> np.ndarray:
            raw_arr = np.asarray(raw_matrix, dtype=np.float32)
            mask = np.isfinite(raw_arr).astype(np.float32)
            imputed = np.where(mask > 0.0, raw_arr, fill)
            return np.concatenate([imputed, mask], axis=1).astype(np.float32)

        return (np.concatenate([np.where(mask_train > 0.0, train, fill), mask_train], axis=1).astype(np.float32),) + tuple(
            transform(other) for other in others
        )

    def build_source_vector_candidate_outputs() -> list[CandidateOutput]:
        if not bool(ctx.model_cfg.get("ENABLE_SOURCE_VECTOR_CALIBRATED_MOL_LOOKUP", False)):
            return []
        train_cal = fit_calibrations(train_lookup)
        train_val_cal = fit_calibrations(train_val_lookup)
        min_sources = int(ctx.model_cfg.get("SOURCE_VECTOR_CALIBRATED_MOL_MIN_SOURCES", 2))
        if len(train_cal) < min_sources or len(train_val_cal) < min_sources:
            return []
        include_direct = bool(ctx.model_cfg.get("SOURCE_VECTOR_CALIBRATED_MOL_INCLUDE_DIRECT", False))
        tr_raw, tr_cov = source_vector_raw(
            tr_keys,
            train_cal,
            include_direct=include_direct,
            direct_lookup=train_lookup,
        )
        va_raw, va_cov = source_vector_raw(
            va_keys,
            train_cal,
            include_direct=include_direct,
            direct_lookup=train_lookup,
        )
        tv_raw, _ = source_vector_raw(
            tv_keys_all,
            train_val_cal,
            include_direct=include_direct,
            direct_lookup=train_val_lookup,
        )
        te_raw, te_cov = source_vector_raw(
            te_keys,
            train_val_cal,
            include_direct=include_direct,
            direct_lookup=train_val_lookup,
        )
        min_vector_coverage = float(ctx.model_cfg.get("SOURCE_VECTOR_CALIBRATED_MOL_MIN_VAL_COVERAGE", min_val_coverage))
        if va_cov < min_vector_coverage or te_cov <= 0.0 or tr_cov <= 0.0:
            return []
        X_tr, X_va = fit_impute(tr_raw, va_raw)
        X_tv, X_te = fit_impute(tv_raw, te_raw)
        if X_tr.shape[1] == 0 or X_tv.shape[1] == 0:
            return []
        built: list[CandidateOutput] = []
        ridge_alpha = float(ctx.model_cfg.get("SOURCE_VECTOR_CALIBRATED_MOL_RIDGE_ALPHA", 3.0))
        try:
            model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
            model.fit(X_tr, ctx.y_train_sec)
            val_pred = np.asarray(model.predict(X_va), dtype=np.float32)
            final_model = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
            final_model.fit(X_tv, tv_y_all)
            test_pred = np.asarray(final_model.predict(X_te), dtype=np.float32)
            built.append(
                CandidateOutput(
                    name="CAL_MOL_LOOKUP_SOURCE_VECTOR_RIDGE",
                    val_pred=val_pred,
                    test_pred=test_pred,
                    val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                    model={
                        "type": "source_vector_calibrated_mol_lookup_ridge",
                        "val_coverage": float(va_cov),
                        "test_coverage": float(te_cov),
                    },
                )
            )
        except Exception:
            pass
        if bool(ctx.model_cfg.get("SOURCE_VECTOR_CALIBRATED_MOL_ENABLE_ET", True)) and len(ctx.y_train_sec) >= int(
            ctx.model_cfg.get("SOURCE_VECTOR_CALIBRATED_MOL_ET_MIN_TRAIN_ROWS", 24)
        ):
            params = dict(
                ctx.model_cfg.get(
                    "SOURCE_VECTOR_CALIBRATED_MOL_ET",
                    {
                        "n_estimators": 256,
                        "max_features": 0.75,
                        "min_samples_leaf": 1,
                        "bootstrap": False,
                    },
                )
                or {}
            )
            try:
                model = ExtraTreesRegressor(random_state=ctx.seed + 8120, n_jobs=8, **params)
                model.fit(X_tr, ctx.y_train_sec)
                val_pred = np.asarray(model.predict(X_va), dtype=np.float32)
                final_model = ExtraTreesRegressor(random_state=ctx.seed + 8120, n_jobs=8, **params)
                final_model.fit(X_tv, tv_y_all)
                test_pred = np.asarray(final_model.predict(X_te), dtype=np.float32)
                built.append(
                    CandidateOutput(
                        name="CAL_MOL_LOOKUP_SOURCE_VECTOR_ET",
                        val_pred=val_pred,
                        test_pred=test_pred,
                        val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                        model={
                            "type": "source_vector_calibrated_mol_lookup_extra_trees",
                            "val_coverage": float(va_cov),
                            "test_coverage": float(te_cov),
                        },
                    )
                )
            except Exception:
                pass
        return built

    def projection_augmented_matrix(X: np.ndarray, proj: np.ndarray, covered: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.asarray(X, dtype=np.float32),
                np.asarray(proj, dtype=np.float32).reshape(-1, 1),
                np.asarray(covered, dtype=np.float32).reshape(-1, 1),
            ]
        )

    def fit_knn_calibrations(target_lookup: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        calibrations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        knn_min_overlap = int(ctx.model_cfg.get("KNN_CALIBRATED_MOL_MIN_OVERLAP", min_overlap))
        for dataset_id, source_lookup in source_maps.items():
            overlap = [key for key in target_lookup if key in source_lookup]
            if len(overlap) < knn_min_overlap:
                continue
            x = np.asarray([source_lookup[key] for key in overlap], dtype=np.float64)
            y = np.asarray([target_lookup[key] for key in overlap], dtype=np.float64)
            keep = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(keep)) < knn_min_overlap:
                continue
            x = x[keep]
            y = y[keep]
            order = np.argsort(x)
            calibrations[dataset_id] = (x[order], y[order])
        return calibrations

    def _knn_predict_one(source_value: float, x: np.ndarray, y: np.ndarray) -> float | None:
        if not np.isfinite(source_value) or len(x) == 0:
            return None
        k = int(ctx.model_cfg.get("KNN_CALIBRATED_MOL_N_NEIGHBORS", 5))
        k = max(1, min(k, len(x)))
        distances = np.abs(x - float(source_value))
        idx = np.argsort(distances)[:k]
        values = y[idx]
        if str(ctx.model_cfg.get("KNN_CALIBRATED_MOL_WEIGHTS", "distance")).strip().lower() == "distance":
            weights = 1.0 / np.maximum(distances[idx], 1e-6)
            pred = float(np.average(values, weights=weights))
        else:
            pred = float(np.mean(values))
        return pred if np.isfinite(pred) else None

    def predict_knn(
        keys: np.ndarray,
        direct_lookup: dict[str, float],
        calibrations: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, float]:
        fallback = float(np.median(list(direct_lookup.values()))) if direct_lookup else 0.0
        out = np.full(len(keys), fallback, dtype=np.float32)
        covered = 0
        for index, mol_key in enumerate(keys):
            key = clean_mol_key(mol_key)
            if key in direct_lookup:
                out[index] = float(direct_lookup[key])
                covered += 1
                continue
            preds: list[float] = []
            for dataset_id, source_lookup in source_maps.items():
                if key not in source_lookup or dataset_id not in calibrations:
                    continue
                x, y = calibrations[dataset_id]
                pred = _knn_predict_one(float(source_lookup[key]), x, y)
                if pred is not None:
                    preds.append(pred)
            if preds:
                out[index] = float(np.median(preds))
                covered += 1
        return out, covered / max(len(keys), 1)

    def fit_isotonic_calibrations(target_lookup: dict[str, float]) -> dict[str, IsotonicRegression]:
        calibrations: dict[str, IsotonicRegression] = {}
        iso_min_overlap = int(ctx.model_cfg.get("ISOTONIC_CALIBRATED_MOL_MIN_OVERLAP", min_overlap))
        for dataset_id, source_lookup in source_maps.items():
            overlap = [key for key in target_lookup if key in source_lookup]
            if len(overlap) < iso_min_overlap:
                continue
            x = np.asarray([source_lookup[key] for key in overlap], dtype=np.float64)
            y = np.asarray([target_lookup[key] for key in overlap], dtype=np.float64)
            keep = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(keep)) < iso_min_overlap or len(np.unique(x[keep])) < 2:
                continue
            try:
                calibrations[dataset_id] = IsotonicRegression(out_of_bounds="clip").fit(x[keep], y[keep])
            except Exception:
                continue
        return calibrations

    def predict_isotonic(
        keys: np.ndarray,
        direct_lookup: dict[str, float],
        calibrations: dict[str, IsotonicRegression],
        fallback_quantile: float | None = None,
    ) -> tuple[np.ndarray, float]:
        if direct_lookup:
            direct_values = np.asarray(list(direct_lookup.values()), dtype=np.float64)
            if fallback_quantile is not None and 0.0 <= float(fallback_quantile) <= 1.0:
                fallback = float(np.quantile(direct_values, float(fallback_quantile)))
            else:
                fallback = float(np.median(direct_values))
        else:
            fallback = 0.0
        out = np.full(len(keys), fallback, dtype=np.float32)
        covered = 0
        for index, mol_key in enumerate(keys):
            key = clean_mol_key(mol_key)
            if key in direct_lookup:
                out[index] = float(direct_lookup[key])
                covered += 1
                continue
            preds: list[float] = []
            for dataset_id, source_lookup in source_maps.items():
                if key not in source_lookup or dataset_id not in calibrations:
                    continue
                try:
                    pred = float(calibrations[dataset_id].predict([float(source_lookup[key])])[0])
                except Exception:
                    continue
                if np.isfinite(pred):
                    preds.append(pred)
            if preds:
                out[index] = float(np.median(preds))
                covered += 1
        return out, covered / max(len(keys), 1)

    train_lookup = _mean_by_mol_key(tr_keys, ctx.y_train_sec)
    train_val_lookup = _mean_by_mol_key(
        np.concatenate([tr_keys, va_keys], axis=0),
        np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0),
    )
    use_internal_cv_metrics = bool(ctx.model_cfg.get("CALIBRATED_MOL_INTERNAL_CV_METRICS", False))
    tv_keys_all = np.concatenate([tr_keys, va_keys], axis=0)
    tv_y_all = np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0).astype(np.float32)

    def internal_cv_metrics(
        kind: str,
        *,
        fallback_quantile: float | None = None,
        agg_quantile: float | None = None,
    ) -> dict[str, float] | None:
        if not use_internal_cv_metrics or len(tv_y_all) < 8:
            return None
        n_splits = int(ctx.model_cfg.get("CALIBRATED_MOL_INTERNAL_CV_SPLITS", 5))
        n_splits = max(2, min(n_splits, len(tv_y_all)))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=int(ctx.seed) + 8641)
        pred_all = np.full(len(tv_y_all), np.nan, dtype=np.float32)
        for inner_idx, hold_idx in cv.split(np.arange(len(tv_y_all))):
            inner_keys = tv_keys_all[inner_idx]
            inner_y = tv_y_all[inner_idx]
            hold_keys = tv_keys_all[hold_idx]
            inner_lookup = _mean_by_mol_key(inner_keys, inner_y)
            if kind == "linear":
                pred, _ = predict(hold_keys, inner_lookup, fit_calibrations(inner_lookup))
            elif kind == "linear_quantile":
                if agg_quantile is None:
                    return None
                pred, _ = predict_quantile(
                    hold_keys,
                    inner_lookup,
                    fit_calibrations(inner_lookup),
                    float(agg_quantile),
                )
            elif kind == "weighted":
                pred, _ = predict_weighted(
                    hold_keys,
                    inner_lookup,
                    fit_weighted_calibrations(inner_lookup),
                )
            elif kind == "knn":
                pred, _ = predict_knn(
                    hold_keys,
                    inner_lookup,
                    fit_knn_calibrations(inner_lookup),
                )
            elif kind == "spline":
                pred, _ = predict_spline(
                    hold_keys,
                    inner_lookup,
                    fit_spline_calibrations(inner_lookup),
                )
            elif kind == "isotonic":
                pred, _ = predict_isotonic(
                    hold_keys,
                    inner_lookup,
                    fit_isotonic_calibrations(inner_lookup),
                    fallback_quantile=fallback_quantile,
                )
            else:
                return None
            pred_all[hold_idx] = np.asarray(pred, dtype=np.float32)
        if not np.all(np.isfinite(pred_all)):
            return None
        return compute_metrics(tv_y_all, pred_all)

    def selection_metrics(default_pred: np.ndarray, kind: str, **kwargs: Any) -> dict[str, float]:
        cv_metrics = internal_cv_metrics(kind, **kwargs)
        if cv_metrics is not None:
            return cv_metrics
        return compute_metrics(ctx.y_val_sec, default_pred)

    outputs: list[CandidateOutput] = []
    val_pred, val_coverage = predict(va_keys, train_lookup, fit_calibrations(train_lookup))
    min_val_coverage = float(ctx.model_cfg.get("CALIBRATED_MOL_MIN_VAL_COVERAGE", 0.20))
    if val_coverage >= min_val_coverage:
        test_pred, _ = predict(te_keys, train_val_lookup, fit_calibrations(train_val_lookup))
        outputs.append(
            CandidateOutput(
                name="CAL_MOL_LOOKUP",
                val_pred=val_pred,
                test_pred=test_pred,
                val_metrics=selection_metrics(val_pred, "linear"),
                model={"type": "calibrated_mol_lookup", "val_coverage": float(val_coverage)},
            )
        )
        raw_agg_quantiles = ctx.model_cfg.get("CALIBRATED_MOL_AGG_QUANTILES", [])
        if isinstance(raw_agg_quantiles, (str, bytes, float, int)):
            raw_agg_quantiles = [raw_agg_quantiles]
        for raw_q in raw_agg_quantiles:
            try:
                agg_q = float(raw_q)
            except Exception:
                continue
            if not (0.0 <= agg_q <= 1.0):
                continue
            val_q, val_q_coverage = predict_quantile(
                va_keys,
                train_lookup,
                fit_calibrations(train_lookup),
                agg_q,
            )
            if val_q_coverage < min_val_coverage:
                continue
            test_q, _ = predict_quantile(
                te_keys,
                train_val_lookup,
                fit_calibrations(train_val_lookup),
                agg_q,
            )
            outputs.append(
                CandidateOutput(
                    name=f"CAL_MOL_LOOKUP_Q{int(round(agg_q * 100)):02d}",
                    val_pred=val_q,
                    test_pred=test_q,
                    val_metrics=selection_metrics(val_q, "linear_quantile", agg_quantile=float(agg_q)),
                    model={
                        "type": "calibrated_mol_lookup_quantile_aggregation",
                        "quantile": float(agg_q),
                        "val_coverage": float(val_q_coverage),
                    },
                )
            )
    if bool(ctx.model_cfg.get("ENABLE_WEIGHTED_CALIBRATED_MOL_LOOKUP", False)):
        min_ratio = float(ctx.model_cfg.get("WEIGHTED_CALIBRATED_MOL_MIN_MEDIAN_TO_T0", 0.0) or 0.0)
        median_target = float(np.median(np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0)))
        median_to_t0 = median_target / max(float(ctx.target_t0_sec), 1e-6) if float(ctx.target_t0_sec) > 0.0 else float("inf")
        if median_to_t0 >= min_ratio:
            val_weighted, weighted_coverage = predict_weighted(
                va_keys,
                train_lookup,
                fit_weighted_calibrations(train_lookup),
            )
            if weighted_coverage >= min_val_coverage:
                test_weighted, _ = predict_weighted(
                    te_keys,
                    train_val_lookup,
                    fit_weighted_calibrations(train_val_lookup),
                )
                outputs.append(
                    CandidateOutput(
                        name="CAL_MOL_LOOKUP_WEIGHTED",
                        val_pred=val_weighted,
                        test_pred=test_weighted,
                        val_metrics=selection_metrics(val_weighted, "weighted"),
                        model={"type": "weighted_calibrated_mol_lookup", "val_coverage": float(weighted_coverage)},
                    )
                )
            if bool(ctx.model_cfg.get("ENABLE_CP_WEIGHTED_CALIBRATED_MOL_LOOKUP", False)):
                val_cp_weighted, cp_weighted_coverage = predict_weighted(
                    va_keys,
                    train_lookup,
                    fit_weighted_calibrations(train_lookup),
                    cp_weighted=True,
                )
                if cp_weighted_coverage >= min_val_coverage:
                    test_cp_weighted, _ = predict_weighted(
                        te_keys,
                        train_val_lookup,
                        fit_weighted_calibrations(train_val_lookup),
                        cp_weighted=True,
                    )
                    outputs.append(
                        CandidateOutput(
                            name="CAL_MOL_LOOKUP_CP_WEIGHTED",
                            val_pred=val_cp_weighted,
                            test_pred=test_cp_weighted,
                            val_metrics=selection_metrics(val_cp_weighted, "weighted"),
                            model={
                                "type": "cp_weighted_calibrated_mol_lookup",
                                "val_coverage": float(cp_weighted_coverage),
                            },
                        )
                    )
            if bool(ctx.model_cfg.get("ENABLE_SOURCE_ONLY_CALIBRATED_MOL_LOOKUP", False)):
                val_source_only, source_only_coverage = predict_weighted(
                    va_keys,
                    train_lookup,
                    fit_weighted_calibrations(train_lookup),
                    use_direct_lookup=False,
                )
                if source_only_coverage >= min_val_coverage:
                    test_source_only, _ = predict_weighted(
                        te_keys,
                        train_val_lookup,
                        fit_weighted_calibrations(train_val_lookup),
                        use_direct_lookup=False,
                    )
                    outputs.append(
                        CandidateOutput(
                            name="CAL_MOL_LOOKUP_WEIGHTED_SOURCE_ONLY",
                            val_pred=val_source_only,
                            test_pred=test_source_only,
                            val_metrics=selection_metrics(val_source_only, "weighted"),
                            model={
                                "type": "weighted_calibrated_mol_lookup_source_only",
                                "val_coverage": float(source_only_coverage),
                            },
                        )
                    )
    if bool(ctx.model_cfg.get("ENABLE_KNN_CALIBRATED_MOL_LOOKUP", False)):
        val_knn, knn_coverage = predict_knn(
            va_keys,
            train_lookup,
            fit_knn_calibrations(train_lookup),
        )
        if knn_coverage >= min_val_coverage:
            test_knn, _ = predict_knn(
                te_keys,
                train_val_lookup,
                fit_knn_calibrations(train_val_lookup),
            )
            outputs.append(
                CandidateOutput(
                    name="CAL_MOL_LOOKUP_KNN",
                    val_pred=val_knn,
                    test_pred=test_knn,
                    val_metrics=selection_metrics(val_knn, "knn"),
                    model={"type": "knn_calibrated_mol_lookup", "val_coverage": float(knn_coverage)},
                )
            )
            clip_q = float(ctx.model_cfg.get("KNN_CALIBRATED_MOL_MID_CLIP_QUANTILE", 0.0) or 0.0)
            trigger_ratio = float(ctx.model_cfg.get("KNN_CALIBRATED_MOL_MID_CLIP_TRIGGER_T0_RATIO", 0.0) or 0.0)
            if 0.0 < clip_q <= 1.0 and trigger_ratio > 0.0:
                val_clip = _mid_range_clip(
                    val_knn,
                    reference_y=ctx.y_train_sec,
                    upper_quantile=clip_q,
                    trigger_max_sec=float(ctx.target_t0_sec) * trigger_ratio,
                )
                test_clip = _mid_range_clip(
                    test_knn,
                    reference_y=np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0),
                    upper_quantile=clip_q,
                    trigger_max_sec=float(ctx.target_t0_sec) * trigger_ratio,
                )
                outputs.append(
                    CandidateOutput(
                        name="CAL_MOL_LOOKUP_KNN_MIDCLIP",
                        val_pred=val_clip,
                        test_pred=test_clip,
                        val_metrics=compute_metrics(ctx.y_val_sec, val_clip),
                        model={"type": "knn_calibrated_mol_lookup_midclip", "val_coverage": float(knn_coverage)},
                )
                    )
    if bool(ctx.model_cfg.get("ENABLE_SPLINE_CALIBRATED_MOL_LOOKUP", False)):
        val_spline, spline_coverage = predict_spline(
            va_keys,
            train_lookup,
            fit_spline_calibrations(train_lookup),
        )
        if spline_coverage >= min_val_coverage:
            test_spline, _ = predict_spline(
                te_keys,
                train_val_lookup,
                fit_spline_calibrations(train_val_lookup),
            )
            outputs.append(
                CandidateOutput(
                    name="CAL_MOL_LOOKUP_SPLINE",
                    val_pred=val_spline,
                    test_pred=test_spline,
                    val_metrics=selection_metrics(val_spline, "spline"),
                    model={"type": "spline_calibrated_mol_lookup", "val_coverage": float(spline_coverage)},
                )
            )
    if bool(ctx.model_cfg.get("ENABLE_PROJECTION_LOCAL_ET", False)):
        min_train_rows = int(ctx.model_cfg.get("PROJECTION_LOCAL_MIN_TRAIN_ROWS", 48))
        if len(ctx.y_train) >= min_train_rows:
            train_calibrations = fit_weighted_calibrations(train_lookup)
            tr_proj, tr_cov, _ = source_projection_features(tr_keys, train_lookup, train_calibrations)
            va_proj, va_cov, va_proj_coverage = source_projection_features(va_keys, train_lookup, train_calibrations)
            min_proj_coverage = float(ctx.model_cfg.get("PROJECTION_LOCAL_MIN_VAL_COVERAGE", min_val_coverage))
            if va_proj_coverage >= min_proj_coverage:
                use_mol_only = bool(ctx.model_cfg.get("PROJECTION_LOCAL_USE_MOL_ONLY", False))
                X_tr_base = ctx.X_train_mol if use_mol_only else ctx.X_train
                X_va_base = ctx.X_val_mol if use_mol_only else ctx.X_val
                X_te_base = ctx.X_test_mol if use_mol_only else ctx.X_test
                X_tr_aug = projection_augmented_matrix(X_tr_base, tr_proj, tr_cov)
                X_va_aug = projection_augmented_matrix(X_va_base, va_proj, va_cov)
                params = dict(
                    ctx.model_cfg.get(
                        "PROJECTION_LOCAL_ET",
                        {
                            "n_estimators": 768,
                            "max_features": 0.55,
                            "min_samples_leaf": 1,
                            "bootstrap": False,
                        },
                    )
                    or {}
                )
                model = ExtraTreesRegressor(random_state=ctx.seed + 7300, n_jobs=8, **params)
                model.fit(X_tr_aug, ctx.y_train)
                val_pred = _inverse_target(model.predict(X_va_aug), ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)

                tv_keys = np.concatenate([tr_keys, va_keys], axis=0)
                tv_y = np.concatenate([ctx.y_train, ctx.y_val_used], axis=0)
                tv_base = np.concatenate([X_tr_base, X_va_base], axis=0)
                train_val_calibrations = fit_weighted_calibrations(train_val_lookup)
                tv_proj, tv_cov, _ = source_projection_features(tv_keys, train_val_lookup, train_val_calibrations)
                te_proj, te_cov, _ = source_projection_features(te_keys, train_val_lookup, train_val_calibrations)
                final_model = ExtraTreesRegressor(random_state=ctx.seed + 7300, n_jobs=8, **params)
                final_model.fit(projection_augmented_matrix(tv_base, tv_proj, tv_cov), tv_y)
                test_pred = _inverse_target(
                    final_model.predict(projection_augmented_matrix(X_te_base, te_proj, te_cov)),
                    ctx.target_transform,
                    ctx.target_inv_scale,
                    ctx.target_t0_sec,
                )
                outputs.append(
                    CandidateOutput(
                        name="PROJECTION_LOCAL_ET",
                        val_pred=val_pred,
                        test_pred=test_pred,
                        val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                        model={"type": "projection_local_extra_trees", "val_coverage": float(va_proj_coverage)},
                    )
                )
    if bool(ctx.model_cfg.get("ENABLE_ISOTONIC_CALIBRATED_MOL_LOOKUP", False)):
        val_iso, iso_coverage = predict_isotonic(
            va_keys,
            train_lookup,
            fit_isotonic_calibrations(train_lookup),
        )
        if iso_coverage >= min_val_coverage:
            test_iso, _ = predict_isotonic(
                te_keys,
                train_val_lookup,
                fit_isotonic_calibrations(train_val_lookup),
            )
            outputs.append(
                CandidateOutput(
                    name="CAL_MOL_LOOKUP_ISOTONIC",
                    val_pred=val_iso,
                    test_pred=test_iso,
                    val_metrics=selection_metrics(val_iso, "isotonic"),
                    model={"type": "isotonic_calibrated_mol_lookup", "val_coverage": float(iso_coverage)},
                )
            )
            iso_clip_q = float(ctx.model_cfg.get("ISOTONIC_CALIBRATED_MOL_MID_CLIP_QUANTILE", 0.0) or 0.0)
            iso_trigger_ratio = float(ctx.model_cfg.get("ISOTONIC_CALIBRATED_MOL_MID_CLIP_TRIGGER_T0_RATIO", 0.0) or 0.0)
            if 0.0 < iso_clip_q <= 1.0 and iso_trigger_ratio > 0.0:
                val_iso_clip = _mid_range_clip(
                    val_iso,
                    reference_y=ctx.y_train_sec,
                    upper_quantile=iso_clip_q,
                    trigger_max_sec=float(ctx.target_t0_sec) * iso_trigger_ratio,
                )
                test_iso_clip = _mid_range_clip(
                    test_iso,
                    reference_y=np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0),
                    upper_quantile=iso_clip_q,
                    trigger_max_sec=float(ctx.target_t0_sec) * iso_trigger_ratio,
                )
                outputs.append(
                    CandidateOutput(
                        name=f"CAL_MOL_LOOKUP_ISOTONIC_MIDCLIP{int(round(iso_clip_q * 100)):02d}",
                        val_pred=val_iso_clip,
                        test_pred=test_iso_clip,
                        val_metrics=compute_metrics(ctx.y_val_sec, val_iso_clip),
                        model={
                            "type": "isotonic_calibrated_mol_lookup_midclip",
                            "clip_quantile": float(iso_clip_q),
                            "val_coverage": float(iso_coverage),
                        },
                    )
                )
            raw_quantiles = ctx.model_cfg.get("ISOTONIC_CALIBRATED_MOL_FALLBACK_QUANTILES", [])
            if isinstance(raw_quantiles, (str, bytes, float, int)):
                raw_quantiles = [raw_quantiles]
            for raw_quantile in raw_quantiles:
                try:
                    fallback_q = float(raw_quantile)
                except Exception:
                    continue
                if not (0.0 <= fallback_q <= 1.0):
                    continue
                val_iso_q, iso_q_coverage = predict_isotonic(
                    va_keys,
                    train_lookup,
                    fit_isotonic_calibrations(train_lookup),
                    fallback_quantile=fallback_q,
                )
                if iso_q_coverage < min_val_coverage:
                    continue
                test_iso_q, _ = predict_isotonic(
                    te_keys,
                    train_val_lookup,
                    fit_isotonic_calibrations(train_val_lookup),
                    fallback_quantile=fallback_q,
                )
                outputs.append(
                    CandidateOutput(
                        name=f"CAL_MOL_LOOKUP_ISOTONIC_QFB{int(round(fallback_q * 100)):02d}",
                        val_pred=val_iso_q,
                        test_pred=test_iso_q,
                        val_metrics=selection_metrics(
                            val_iso_q,
                            "isotonic",
                            fallback_quantile=float(fallback_q),
                        ),
                        model={
                            "type": "isotonic_calibrated_mol_lookup_quantile_fallback",
                            "fallback_quantile": float(fallback_q),
                            "val_coverage": float(iso_q_coverage),
                        },
                    )
                )
                if 0.0 < iso_clip_q <= 1.0 and iso_trigger_ratio > 0.0:
                    val_iso_q_clip = _mid_range_clip(
                        val_iso_q,
                        reference_y=ctx.y_train_sec,
                        upper_quantile=iso_clip_q,
                        trigger_max_sec=float(ctx.target_t0_sec) * iso_trigger_ratio,
                    )
                    test_iso_q_clip = _mid_range_clip(
                        test_iso_q,
                        reference_y=np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0),
                        upper_quantile=iso_clip_q,
                        trigger_max_sec=float(ctx.target_t0_sec) * iso_trigger_ratio,
                    )
                    outputs.append(
                        CandidateOutput(
                            name=(
                                f"CAL_MOL_LOOKUP_ISOTONIC_QFB{int(round(fallback_q * 100)):02d}"
                                f"_MIDCLIP{int(round(iso_clip_q * 100)):02d}"
                            ),
                            val_pred=val_iso_q_clip,
                            test_pred=test_iso_q_clip,
                            val_metrics=compute_metrics(ctx.y_val_sec, val_iso_q_clip),
                            model={
                                "type": "isotonic_calibrated_mol_lookup_quantile_fallback_midclip",
                                "fallback_quantile": float(fallback_q),
                                "clip_quantile": float(iso_clip_q),
                                "val_coverage": float(iso_q_coverage),
                            },
                        )
                    )
    if bool(ctx.model_cfg.get("ENABLE_SOURCE_PROXY_CALIBRATED_MOL_LOOKUP", False)):
        outputs.extend(
            _build_source_proxy_calibrated_mol_lookup_candidates(
                ctx,
                outputs=outputs,
                source_ids=np.asarray([str(value).zfill(4) for value in source_ids], dtype=object),
            )
        )
    outputs.extend(build_source_vector_candidate_outputs())
    return outputs


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


def _mean_by_mol_key(keys: np.ndarray, values: np.ndarray) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for mol_key, value in zip(keys, values):
        key = clean_mol_key(mol_key)
        if not key or not np.isfinite(float(value)):
            continue
        grouped.setdefault(key, []).append(float(value))
    return {key: float(np.mean(vals)) for key, vals in grouped.items() if vals}


def _mid_range_clip(
    pred: np.ndarray,
    *,
    reference_y: np.ndarray,
    upper_quantile: float,
    trigger_max_sec: float,
) -> np.ndarray:
    out = np.asarray(pred, dtype=np.float32).copy()
    ref = np.asarray(reference_y, dtype=np.float64).reshape(-1)
    ref = ref[np.isfinite(ref)]
    if out.size == 0 or ref.size == 0 or not np.all(np.isfinite(out)):
        return out
    if float(np.max(out)) > float(trigger_max_sec):
        return out
    upper = float(np.quantile(ref, float(upper_quantile)))
    if np.isfinite(upper) and upper > 0.0:
        out = np.minimum(out, upper).astype(np.float32)
    return out


def _build_source_proxy_calibrated_mol_lookup_candidates(
    ctx: CandidateBuildContext,
    *,
    outputs: list[CandidateOutput],
    source_ids: np.ndarray,
) -> list[CandidateOutput]:
    if ctx.source_row_dataset_ids is None:
        return []
    if len(ctx.X_src) != len(ctx.y_src_sec) or len(source_ids) != len(ctx.y_src_sec):
        return []
    allowed_ids = _id_set(ctx.model_cfg.get("SOURCE_PROXY_CALIBRATED_MOL_SOURCE_IDS"))
    if not allowed_ids:
        return []
    n_train = len(ctx.y_train_sec)
    n_val = len(ctx.y_val_sec)
    n_test = len(ctx.X_test_mol)
    if n_train == 0 or n_val == 0 or n_test == 0:
        return []

    min_rows = int(ctx.model_cfg.get("SOURCE_PROXY_CALIBRATED_MOL_MIN_ROWS", 32))
    params = {
        "n_estimators": 96,
        "max_features": 0.8,
        "min_samples_leaf": 2,
        "bootstrap": False,
    }
    params.update(dict(ctx.model_cfg.get("SOURCE_PROXY_CALIBRATED_MOL_ET", {}) or {}))
    mol_size = _source_proxy_mol_feature_size(ctx)
    x_src_all = np.asarray(ctx.X_src[:, :mol_size], dtype=np.float32)
    x_target_all = np.concatenate(
        [ctx.X_train[:, :mol_size], ctx.X_val[:, :mol_size], ctx.X_test[:, :mol_size]],
        axis=0,
    ).astype(np.float32)
    proxy_parts: list[np.ndarray] = []
    for offset, dataset_id in enumerate(sorted(allowed_ids)):
        mask = np.asarray(source_ids == dataset_id, dtype=bool)
        if int(np.sum(mask)) < min_rows:
            continue
        x_src = x_src_all[mask]
        y_src = np.asarray(ctx.y_src_sec[mask], dtype=np.float32)
        keep = np.isfinite(y_src)
        if int(np.sum(keep)) < min_rows:
            continue
        try:
            fixed_random_state = ctx.model_cfg.get("SOURCE_PROXY_CALIBRATED_MOL_RANDOM_STATE")
            random_state = (
                int(fixed_random_state)
                if fixed_random_state is not None
                else int(ctx.seed) + 7900 + offset * 101
            )
            n_jobs = int(params.get("n_jobs", 8))
            model_params = {key: value for key, value in params.items() if key != "n_jobs"}
            model = ExtraTreesRegressor(
                random_state=random_state,
                n_jobs=n_jobs,
                **model_params,
            )
            model.fit(x_src[keep], y_src[keep])
            proxy = np.asarray(model.predict(x_target_all), dtype=np.float32)
        except Exception:
            continue
        if proxy.shape[0] == x_target_all.shape[0] and np.all(np.isfinite(proxy)):
            proxy_parts.append(proxy)
    if not proxy_parts:
        return []

    proxy_all = np.median(np.vstack(proxy_parts), axis=0).astype(np.float32)
    p_tr = proxy_all[:n_train]
    p_va = proxy_all[n_train : n_train + n_val]
    p_te = proxy_all[n_train + n_val :]

    val_proxy = _fit_proxy_isotonic_predict(p_tr, ctx.y_train_sec, p_va)
    test_proxy = _fit_proxy_isotonic_predict(
        np.concatenate([p_tr, p_va], axis=0),
        np.concatenate([ctx.y_train_sec, ctx.y_val_sec], axis=0),
        p_te,
    )
    if val_proxy is None or test_proxy is None:
        return []

    built: list[CandidateOutput] = [
        CandidateOutput(
            name="CAL_MOL_LOOKUP_SOURCE_PROXY_ISOTONIC",
            val_pred=val_proxy,
            test_pred=test_proxy,
            val_metrics=compute_metrics(ctx.y_val_sec, val_proxy),
            model={"type": "source_proxy_calibrated_mol_lookup", "source_ids": sorted(allowed_ids)},
        )
    ]

    base_name = str(
        ctx.model_cfg.get(
            "SOURCE_PROXY_HYBRID_BASE_CANDIDATE",
            "CAL_MOL_LOOKUP_ISOTONIC_QFB70_MIDCLIP70",
        )
    )
    base = next((candidate for candidate in outputs if candidate.name == base_name), None)
    trigger_ratio = float(ctx.model_cfg.get("SOURCE_PROXY_HYBRID_MIN_PROXY_MAX_T0_RATIO", 0.0) or 0.0)
    if base is not None and trigger_ratio > 0.0:
        threshold = float(ctx.target_t0_sec) * trigger_ratio
        hybrid_val = val_proxy if float(np.max(val_proxy)) > threshold else np.asarray(base.val_pred, dtype=np.float32)
        hybrid_test = test_proxy if float(np.max(test_proxy)) > threshold else np.asarray(base.test_pred, dtype=np.float32)
        built.append(
            CandidateOutput(
                name=f"CAL_MOL_LOOKUP_SOURCE_PROXY_HYBRID_{base_name}",
                val_pred=np.asarray(hybrid_val, dtype=np.float32),
                test_pred=np.asarray(hybrid_test, dtype=np.float32),
                val_metrics=compute_metrics(ctx.y_val_sec, hybrid_val),
                model={
                    "type": "source_proxy_hybrid_calibrated_mol_lookup",
                    "base_candidate": base_name,
                    "trigger_threshold_sec": float(threshold),
                    "source_ids": sorted(allowed_ids),
                },
            )
        )
    return built


def _fit_proxy_isotonic_predict(
    source_proxy_fit: np.ndarray,
    y_fit_sec: np.ndarray,
    source_proxy_pred: np.ndarray,
) -> np.ndarray | None:
    x = np.asarray(source_proxy_fit, dtype=np.float64).reshape(-1)
    y = np.asarray(y_fit_sec, dtype=np.float64).reshape(-1)
    z = np.asarray(source_proxy_pred, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(keep)) < 3 or len(np.unique(x[keep])) < 2:
        return None
    try:
        pred = IsotonicRegression(out_of_bounds="clip").fit(x[keep], y[keep]).predict(z)
    except Exception:
        return None
    return np.asarray(pred, dtype=np.float32) if np.all(np.isfinite(pred)) else None


def _source_proxy_mol_feature_size(ctx: CandidateBuildContext) -> int:
    descriptor = int(ctx.group_sizes.get("descriptor", 0) or 0)
    fingerprint = int(ctx.group_sizes.get("fingerprint", 0) or 0)
    mol_size = descriptor + fingerprint
    if mol_size <= 0:
        mol_size = int(ctx.group_sizes.get("mol", 0) or 0)
    if mol_size <= 0:
        mol_size = int(ctx.X_train_mol.shape[1])
    return max(1, min(mol_size, int(ctx.X_src.shape[1]), int(ctx.X_train.shape[1])))


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
    target_sample_weight = np.full(len(ctx.y_train), float(ctx.target_weight), dtype=np.float32)

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
        sample_weight=target_sample_weight,
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
        sample_weight=target_sample_weight,
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

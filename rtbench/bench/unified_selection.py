from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..metrics import compute_metrics
from .unified_cv import UnifiedCVFold, UnifiedCVFoldPrediction


MetricName = Literal["mae", "medae", "mre", "medre", "rmse", "r2"]
RefitPolicy = Literal["train_only", "train_val"]
SelectionObjective = Literal["pooled_metric", "dataset_balanced_r2"]
EstimatorFactory = Callable[[Mapping[str, Any], int], Any]


@dataclass(frozen=True)
class UnifiedCandidate:
    """One pooled learner/hyperparameter candidate for strict unified selection."""

    name: str
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)
    standardize: bool = True
    estimator_factory: EstimatorFactory | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class UnifiedSelectionResult:
    fold_id: int
    val_pred_sec: np.ndarray
    test_pred_sec: np.ndarray
    model_info: Mapping[str, Any]
    model: Any | None = field(default=None, compare=False, repr=False)


def ridge_candidate_grid(
    alphas: Sequence[float] = (0.1, 1.0, 10.0),
    *,
    prefix: str = "ridge",
    standardize: bool = True,
) -> list[UnifiedCandidate]:
    return [
        UnifiedCandidate(
            name=f"{prefix}[alpha={float(alpha):g}]",
            kind="ridge",
            params={"alpha": float(alpha)},
            standardize=standardize,
        )
        for alpha in alphas
    ]


def elasticnet_candidate_grid(
    *,
    alphas: Sequence[float] = (0.01, 0.1, 1.0),
    l1_ratios: Sequence[float] = (0.1, 0.5, 0.9),
    prefix: str = "elasticnet",
    standardize: bool = True,
    max_iter: int = 10000,
) -> list[UnifiedCandidate]:
    grid = ParameterGrid({"alpha": list(alphas), "l1_ratio": list(l1_ratios)})
    return [
        UnifiedCandidate(
            name=f"{prefix}[alpha={float(params['alpha']):g},l1_ratio={float(params['l1_ratio']):g}]",
            kind="elasticnet",
            params={
                "alpha": float(params["alpha"]),
                "l1_ratio": float(params["l1_ratio"]),
                "max_iter": int(max_iter),
            },
            standardize=standardize,
        )
        for params in grid
    ]


def lgbm_candidate_grid(
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    *,
    prefix: str = "lgbm",
) -> list[UnifiedCandidate]:
    """Return LightGBM candidates without fitting them at import time."""

    grid = ParameterGrid(
        param_grid
        or {
            "num_leaves": [15, 31],
            "learning_rate": [0.03, 0.1],
            "n_estimators": [100],
        }
    )
    candidates: list[UnifiedCandidate] = []
    for i, params in enumerate(grid):
        clean = dict(params)
        label = ",".join(f"{key}={clean[key]}" for key in sorted(clean))
        candidates.append(
            UnifiedCandidate(
                name=f"{prefix}[{label}]" if label else f"{prefix}[{i}]",
                kind="lgbm",
                params=clean,
                standardize=False,
            )
        )
    return candidates


def select_strict_unified_learner(
    fold: UnifiedCVFold,
    candidates: Sequence[UnifiedCandidate],
    *,
    metric: MetricName = "mae",
    objective: SelectionObjective = "pooled_metric",
    refit_policy: RefitPolicy = "train_val",
    random_state: int = 0,
    baseline_metrics: Mapping[str, Mapping[str, float]] | None = None,
    max_mean_dataset_mae: float | None = None,
    mean_dataset_mae_slack: float = 0.0,
    low_dynamic_range_threshold: float = 1e-6,
) -> UnifiedSelectionResult:
    """Select one pooled learner using only pooled train/validation rows.

    The selection path intentionally reads `X_train`, `y_train_sec`, `X_val`,
    and `y_val_sec` only. `X_test` is used after selection for prediction; test
    labels and test metadata are not read by this core selector.
    """

    if not candidates:
        raise ValueError("strict unified selection requires at least one candidate")
    if metric not in {"mae", "medae", "mre", "medre", "rmse", "r2"}:
        raise ValueError(f"unsupported validation metric: {metric!r}")
    if objective not in {"pooled_metric", "dataset_balanced_r2"}:
        raise ValueError(f"unsupported selection objective: {objective!r}")
    if refit_policy not in {"train_only", "train_val"}:
        raise ValueError(f"unsupported refit_policy: {refit_policy!r}")

    _validate_fair_candidates(candidates)

    X_train = np.asarray(fold.X_train, dtype=np.float32)
    y_train = np.asarray(fold.y_train_sec, dtype=np.float32).reshape(-1)
    X_val = np.asarray(fold.X_val, dtype=np.float32)
    y_val = np.asarray(fold.y_val_sec, dtype=np.float32).reshape(-1)
    fold_id = int(getattr(fold, "fold_id", 0))
    val_meta = _validation_meta_for_objective(fold, objective, len(y_val))

    validation_records: list[dict[str, Any]] = []
    candidate_models: list[Any] = []
    candidate_val_preds: list[np.ndarray] = []

    for candidate_index, candidate in enumerate(candidates):
        model = _build_estimator(candidate, int(random_state) + candidate_index)
        model.fit(X_train, y_train)
        val_pred = np.asarray(model.predict(X_val), dtype=np.float32).reshape(-1)
        metrics = compute_metrics(y_val, val_pred)
        record = {
            "candidate_index": int(candidate_index),
            "name": candidate.name,
            "kind": candidate.kind,
            "params": _plain_params(candidate.params),
            "standardize": bool(candidate.standardize),
            "metrics": {key: float(value) for key, value in metrics.items()},
            "selection_score": float(metrics[metric]),
        }
        if objective == "dataset_balanced_r2":
            balanced = _dataset_balanced_validation_metrics(
                y_val,
                val_pred,
                val_meta,
                baseline_metrics=baseline_metrics,
                low_dynamic_range_threshold=float(low_dynamic_range_threshold),
            )
            record["balanced_metrics"] = balanced
            record["selection_score"] = float(balanced["mean_dataset_r2"])
        validation_records.append(record)
        candidate_models.append(model)
        candidate_val_preds.append(val_pred)

    if not validation_records:
        raise RuntimeError("strict unified selection failed to select a candidate")

    selected_index = _select_validation_record_index(
        validation_records,
        metric=metric,
        objective=objective,
        max_mean_dataset_mae=max_mean_dataset_mae,
        mean_dataset_mae_slack=float(mean_dataset_mae_slack),
    )
    best_candidate = candidates[selected_index]
    best_model = candidate_models[selected_index]
    best_val_pred = candidate_val_preds[selected_index]
    if refit_policy == "train_val":
        X_final = np.concatenate([X_train, X_val], axis=0)
        y_final = np.concatenate([y_train, y_val], axis=0)
        final_model = _build_estimator(best_candidate, int(random_state) + selected_index)
        final_model.fit(X_final, y_final)
        final_training_rows = int(len(y_final))
        final_refit_count = 1
    else:
        final_model = best_model
        final_training_rows = int(len(y_train))
        final_refit_count = 0

    X_test = np.asarray(fold.X_test, dtype=np.float32)
    test_pred = np.asarray(final_model.predict(X_test), dtype=np.float32).reshape(-1)
    selected_record = validation_records[selected_index]
    selection_uses = ["X_train", "y_train_sec", "X_val", "y_val_sec"]
    if objective == "dataset_balanced_r2":
        selection_uses.append("val_meta")
    balanced_objective_info = _balanced_objective_info(
        validation_records,
        objective=objective,
        max_mean_dataset_mae=max_mean_dataset_mae,
        mean_dataset_mae_slack=float(mean_dataset_mae_slack),
    )
    model_info = {
        "selector": "strict_unified_pooled_validation"
        if objective == "pooled_metric"
        else "strict_unified_dataset_balanced_validation",
        "fold_id": fold_id,
        "n_model": 1,
        "n_final_models": 1,
        "n_models_per_dataset": 0,
        "candidate_fit_scope": "pooled_train_only",
        "n_pooled_candidate_fits": len(validation_records),
        "final_refit_count": int(final_refit_count),
        "final_fit_scope": "pooled_train" if refit_policy == "train_only" else "pooled_train_val",
        "final_training_rows": final_training_rows,
        "validation_metric": metric,
        "validation_metric_direction": "max" if metric == "r2" else "min",
        "selection_objective": objective,
        "tie_break_policy": "validation metric, then candidate_index order"
        if objective == "pooled_metric"
        else "mean dataset R2, constrained mean dataset MAE, then candidate_index order",
        "refit_policy": refit_policy,
        "selection_uses": selection_uses,
        "prediction_uses": ["X_test"],
        "selection_excludes": [
            "X_test",
            "y_test_sec",
            "test_meta",
            "dataset_indices",
            "test_slices",
            "lookup",
            "HILIC_pool",
            "local_fast",
            "per_dataset_overrides",
        ],
        "candidate_grid": [
            {
                "candidate_index": record["candidate_index"],
                "name": record["name"],
                "kind": record["kind"],
                "params": record["params"],
                "standardize": record["standardize"],
            }
            for record in validation_records
        ],
        "validation_results": validation_records,
        "selected_candidate_index": selected_record["candidate_index"],
        "selected_candidate_name": selected_record["name"],
        "selected_candidate_kind": selected_record["kind"],
        "selected_params": selected_record["params"],
        "selected_validation_metrics": selected_record["metrics"],
    }
    if balanced_objective_info:
        model_info["balanced_objective"] = balanced_objective_info
        model_info["selected_balanced_validation_metrics"] = selected_record.get("balanced_metrics", {})
    return UnifiedSelectionResult(
        fold_id=fold_id,
        val_pred_sec=best_val_pred,
        test_pred_sec=test_pred,
        model_info=model_info,
        model=final_model,
    )


def strict_unified_selector_fit_predict(
    fold: UnifiedCVFold,
    candidates: Sequence[UnifiedCandidate],
    *,
    metric: MetricName = "mae",
    objective: SelectionObjective = "pooled_metric",
    refit_policy: RefitPolicy = "train_val",
    random_state: int = 0,
    baseline_metrics: Mapping[str, Mapping[str, float]] | None = None,
    max_mean_dataset_mae: float | None = None,
    mean_dataset_mae_slack: float = 0.0,
    low_dynamic_range_threshold: float = 1e-6,
) -> UnifiedCVFoldPrediction:
    """Runner-compatible wrapper around `select_strict_unified_learner`."""

    result = select_strict_unified_learner(
        fold,
        candidates,
        metric=metric,
        objective=objective,
        refit_policy=refit_policy,
        random_state=random_state,
        baseline_metrics=baseline_metrics,
        max_mean_dataset_mae=max_mean_dataset_mae,
        mean_dataset_mae_slack=mean_dataset_mae_slack,
        low_dynamic_range_threshold=low_dynamic_range_threshold,
    )
    return UnifiedCVFoldPrediction(
        fold_id=result.fold_id,
        val_pred_sec=result.val_pred_sec,
        test_pred_sec=result.test_pred_sec,
        val_meta=fold.val_meta,
        test_meta=fold.test_meta,
        model_info=result.model_info,
    )


def _build_estimator(candidate: UnifiedCandidate, random_state: int) -> Any:
    if candidate.estimator_factory is not None:
        estimator = candidate.estimator_factory(candidate.params, int(random_state))
    elif candidate.kind == "ridge":
        estimator = Ridge(random_state=int(random_state), **dict(candidate.params))
    elif candidate.kind == "elasticnet":
        estimator = ElasticNet(random_state=int(random_state), **dict(candidate.params))
    elif candidate.kind == "lgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - dependency declared by project
            raise RuntimeError("lightgbm is required for lgbm unified candidates") from exc
        estimator = LGBMRegressor(random_state=int(random_state), verbosity=-1, **dict(candidate.params))
    else:
        raise ValueError(f"unsupported unified candidate kind: {candidate.kind!r}")

    if candidate.standardize and _supports_pipeline_standardization(estimator):
        return make_pipeline(StandardScaler(), estimator)
    return estimator


def _supports_pipeline_standardization(estimator: Any) -> bool:
    return isinstance(estimator, BaseEstimator) or all(hasattr(estimator, attr) for attr in ("fit", "predict"))


def _rank_score(metrics: Mapping[str, float], metric: MetricName) -> float:
    value = float(metrics[metric])
    return -value if metric == "r2" else value


def _validation_meta_for_objective(fold: UnifiedCVFold, objective: SelectionObjective, n_val: int) -> tuple[Any, ...]:
    if objective == "pooled_metric":
        return ()
    val_meta = tuple(getattr(fold, "val_meta"))
    if len(val_meta) != int(n_val):
        raise ValueError(f"validation metadata length mismatch: got={len(val_meta)}, expected={n_val}")
    return val_meta


def _select_validation_record_index(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: MetricName,
    objective: SelectionObjective,
    max_mean_dataset_mae: float | None,
    mean_dataset_mae_slack: float,
) -> int:
    if objective == "pooled_metric":
        return min(
            range(len(records)),
            key=lambda i: (_rank_score(records[i]["metrics"], metric), int(records[i]["candidate_index"])),
        )

    candidate_maes = [
        float(record["balanced_metrics"]["mean_dataset_mae"])
        for record in records
        if np.isfinite(float(record["balanced_metrics"]["mean_dataset_mae"]))
    ]
    if not candidate_maes:
        raise ValueError("dataset-balanced selection could not compute any finite mean dataset MAE values")
    mae_limit = float(max_mean_dataset_mae) if max_mean_dataset_mae is not None else min(candidate_maes) + float(mean_dataset_mae_slack)

    def rank_key(i: int) -> tuple[int, float, float, int]:
        balanced = records[i]["balanced_metrics"]
        mean_mae = float(balanced["mean_dataset_mae"])
        mean_r2 = float(balanced["mean_dataset_r2"])
        feasible = np.isfinite(mean_mae) and mean_mae <= mae_limit
        r2_score = mean_r2 if np.isfinite(mean_r2) else -np.inf
        mae_score = mean_mae if np.isfinite(mean_mae) else np.inf
        return (0 if feasible else 1, -r2_score, mae_score, int(records[i]["candidate_index"]))

    return min(range(len(records)), key=rank_key)


def _balanced_objective_info(
    records: Sequence[Mapping[str, Any]],
    *,
    objective: SelectionObjective,
    max_mean_dataset_mae: float | None,
    mean_dataset_mae_slack: float,
) -> dict[str, Any]:
    if objective != "dataset_balanced_r2":
        return {}
    candidate_maes = [
        float(record["balanced_metrics"]["mean_dataset_mae"])
        for record in records
        if np.isfinite(float(record["balanced_metrics"]["mean_dataset_mae"]))
    ]
    mae_limit = None
    if candidate_maes:
        mae_limit = float(max_mean_dataset_mae) if max_mean_dataset_mae is not None else min(candidate_maes) + float(mean_dataset_mae_slack)
    return {
        "primary": "mean_dataset_r2",
        "mae_constraint": "mean_dataset_mae",
        "max_mean_dataset_mae": mae_limit,
        "mean_dataset_mae_slack": float(mean_dataset_mae_slack),
        "baseline_supplied": any("balanced_metrics" in record and record["balanced_metrics"].get("baseline_dataset_count", 0) for record in records),
    }


def _dataset_balanced_validation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    val_meta: Sequence[Any],
    *,
    baseline_metrics: Mapping[str, Mapping[str, float]] | None,
    low_dynamic_range_threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError(f"validation prediction length mismatch: got={len(y_pred)}, expected={len(y_true)}")
    if len(val_meta) != len(y_true):
        raise ValueError(f"validation metadata length mismatch: got={len(val_meta)}, expected={len(y_true)}")

    groups: dict[str, list[int]] = {}
    for i, row_meta in enumerate(val_meta):
        dataset_id = _dataset_id_from_meta(row_meta)
        groups.setdefault(dataset_id, []).append(i)
    if not groups:
        raise ValueError("dataset-balanced selection requires at least one validation dataset group")

    normalized_baseline = _normalize_baseline_metrics(baseline_metrics)
    dataset_rows: list[dict[str, Any]] = []
    beat_both_count = 0
    beat_mae_count = 0
    beat_r2_count = 0
    baseline_dataset_count = 0
    low_dynamic_range_datasets: list[str] = []

    for dataset_id in sorted(groups):
        idx = np.asarray(groups[dataset_id], dtype=int)
        cur_true = y_true[idx]
        cur_pred = y_pred[idx]
        metrics = compute_metrics(cur_true, cur_pred)
        y_range = float(np.max(cur_true) - np.min(cur_true)) if len(cur_true) else 0.0
        y_std = float(np.std(cur_true)) if len(cur_true) else 0.0
        low_dynamic = bool(len(cur_true) < 2 or y_range <= float(low_dynamic_range_threshold) or y_std <= float(low_dynamic_range_threshold))
        if low_dynamic:
            low_dynamic_range_datasets.append(dataset_id)
        row: dict[str, Any] = {
            "dataset": dataset_id,
            "n_val_rows": int(len(idx)),
            "mae": float(metrics["mae"]),
            "r2": float(metrics["r2"]),
            "rmse": float(metrics["rmse"]),
            "y_range": y_range,
            "y_std": y_std,
            "low_dynamic_range": low_dynamic,
        }
        baseline = normalized_baseline.get(dataset_id)
        if baseline is not None:
            baseline_dataset_count += 1
            baseline_mae = float(baseline["mae"]) if "mae" in baseline else np.nan
            baseline_r2 = float(baseline["r2"]) if "r2" in baseline else np.nan
            beat_mae = bool(np.isfinite(baseline_mae) and row["mae"] < baseline_mae)
            beat_r2 = bool(np.isfinite(baseline_r2) and row["r2"] > baseline_r2)
            beat_both = bool(beat_mae and beat_r2)
            beat_mae_count += int(beat_mae)
            beat_r2_count += int(beat_r2)
            beat_both_count += int(beat_both)
            row.update(
                {
                    "baseline_mae": baseline_mae,
                    "baseline_r2": baseline_r2,
                    "beat_mae": beat_mae,
                    "beat_r2": beat_r2,
                    "beat_both": beat_both,
                }
            )
        dataset_rows.append(row)

    dataset_mae = [float(row["mae"]) for row in dataset_rows if np.isfinite(float(row["mae"]))]
    dataset_r2 = [float(row["r2"]) for row in dataset_rows if np.isfinite(float(row["r2"]))]
    n_datasets = len(dataset_rows)
    return {
        "n_val_rows": int(len(y_true)),
        "n_datasets": int(n_datasets),
        "mean_dataset_mae": float(np.mean(dataset_mae)) if dataset_mae else float("nan"),
        "mean_dataset_r2": float(np.mean(dataset_r2)) if dataset_r2 else float("nan"),
        "baseline_dataset_count": int(baseline_dataset_count),
        "beat_mae_count": int(beat_mae_count),
        "beat_r2_count": int(beat_r2_count),
        "beat_both_count": int(beat_both_count),
        "beat_both_rate": float(beat_both_count / baseline_dataset_count) if baseline_dataset_count else float("nan"),
        "low_dynamic_range_count": int(len(low_dynamic_range_datasets)),
        "low_dynamic_range_datasets": low_dynamic_range_datasets,
        "low_dynamic_range_threshold": float(low_dynamic_range_threshold),
        "datasets": dataset_rows,
    }


def _dataset_id_from_meta(row_meta: Any) -> str:
    if isinstance(row_meta, Mapping):
        raw = row_meta.get("dataset_id", row_meta.get("dataset", None))
    else:
        raw = getattr(row_meta, "dataset_id", getattr(row_meta, "dataset", None))
    if raw is None:
        raise ValueError("validation metadata rows must expose dataset_id or dataset")
    text = str(raw).strip()
    if not text:
        raise ValueError("validation metadata contains an empty dataset id")
    return text.zfill(4)


def _normalize_baseline_metrics(
    baseline_metrics: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, Mapping[str, float]]:
    if baseline_metrics is None:
        return {}
    return {str(dataset_id).strip().zfill(4): values for dataset_id, values in baseline_metrics.items()}


def _validate_fair_candidates(candidates: Sequence[UnifiedCandidate]) -> None:
    for candidate in candidates:
        text = f"{candidate.name} {candidate.kind}".lower()
        for token in ("lookup", "hilic_pool", "local_fast", "local-fast"):
            if token in text:
                raise ValueError(f"candidate {candidate.name!r} violates strict unified fairness constraints: {token}")
        _reject_forbidden_param_keys(candidate.params, path=candidate.name)


def _reject_forbidden_param_keys(value: Any, *, path: str) -> None:
    forbidden = {
        "dataset_override",
        "dataset_overrides",
        "dataset_level_overrides",
        "overrides_by_dataset",
        "per_dataset",
        "per_dataset_override",
        "per_dataset_overrides",
        "lookup",
        "lookup_table",
        "hilic_pool",
        "local_fast",
    }
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in forbidden:
                raise ValueError(f"candidate parameter {path}.{key} violates strict unified fairness constraints")
            _reject_forbidden_param_keys(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, nested in enumerate(value):
            _reject_forbidden_param_keys(nested, path=f"{path}[{i}]")


def _plain_params(params: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            out[str(key)] = value.item()
        elif isinstance(value, np.ndarray):
            out[str(key)] = value.tolist()
        elif isinstance(value, Mapping):
            out[str(key)] = _plain_params(value)
        elif isinstance(value, (list, tuple)):
            out[str(key)] = [item.item() if isinstance(item, np.generic) else item for item in value]
        else:
            out[str(key)] = value
    return out

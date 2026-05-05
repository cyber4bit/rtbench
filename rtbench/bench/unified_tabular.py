from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from .unified_cv import FitPredictFn, UnifiedCVFold, UnifiedCVFoldPrediction


LearnerName = Literal["hist_gradient_boosting", "extra_trees", "random_forest"]
EstimatorFactory = Callable[[Mapping[str, Any], int], Any]


@dataclass(frozen=True)
class PredictionClipConfig:
    enabled: bool = False
    margin_scales: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class UnifiedTabularHeadConfig:
    learner: LearnerName
    params: Mapping[str, Any] = field(default_factory=dict)
    random_state: int = 0
    feature_config: Any | None = None
    clipping: PredictionClipConfig = field(default_factory=PredictionClipConfig)


@dataclass(frozen=True)
class UnifiedTabularFeatureBlock:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    source: str
    n_features: int


def build_hist_gradient_boosting_fit_predict(
    model_cfg: Mapping[str, Any] | None = None,
    *,
    params: Mapping[str, Any] | None = None,
    random_state: int = 0,
    clip_predictions: bool | None = None,
    clip_margin_scales: Sequence[float] | None = None,
) -> FitPredictFn:
    cfg = _head_config(
        "hist_gradient_boosting",
        model_cfg,
        params=params,
        random_state=random_state,
        clip_predictions=clip_predictions,
        clip_margin_scales=clip_margin_scales,
    )
    return _build_tabular_fit_predict(cfg, _hist_gradient_boosting_factory)


def build_extra_trees_fit_predict(
    model_cfg: Mapping[str, Any] | None = None,
    *,
    params: Mapping[str, Any] | None = None,
    random_state: int = 0,
    clip_predictions: bool | None = None,
    clip_margin_scales: Sequence[float] | None = None,
) -> FitPredictFn:
    cfg = _head_config(
        "extra_trees",
        model_cfg,
        params=params,
        random_state=random_state,
        clip_predictions=clip_predictions,
        clip_margin_scales=clip_margin_scales,
    )
    return _build_tabular_fit_predict(cfg, _extra_trees_factory)


def build_random_forest_fit_predict(
    model_cfg: Mapping[str, Any] | None = None,
    *,
    params: Mapping[str, Any] | None = None,
    random_state: int = 0,
    clip_predictions: bool | None = None,
    clip_margin_scales: Sequence[float] | None = None,
) -> FitPredictFn:
    cfg = _head_config(
        "random_forest",
        model_cfg,
        params=params,
        random_state=random_state,
        clip_predictions=clip_predictions,
        clip_margin_scales=clip_margin_scales,
    )
    return _build_tabular_fit_predict(cfg, _random_forest_factory)


def build_unified_tabular_fit_predict(
    model_cfg: Mapping[str, Any] | None = None,
    *,
    learner: str | None = None,
    params: Mapping[str, Any] | None = None,
    random_state: int = 0,
    clip_predictions: bool | None = None,
    clip_margin_scales: Sequence[float] | None = None,
) -> FitPredictFn:
    learner_name = _normalise_learner_name(
        learner
        or _lookup_first(
            model_cfg,
            "UNIFIED_TABULAR_LEARNER",
            "STRICT_UNIFIED_TABULAR_LEARNER",
            "UNIFIED_CV_TABULAR_LEARNER",
            default="hist_gradient_boosting",
        )
    )
    if learner_name == "hist_gradient_boosting":
        return build_hist_gradient_boosting_fit_predict(
            model_cfg,
            params=params,
            random_state=random_state,
            clip_predictions=clip_predictions,
            clip_margin_scales=clip_margin_scales,
        )
    if learner_name == "extra_trees":
        return build_extra_trees_fit_predict(
            model_cfg,
            params=params,
            random_state=random_state,
            clip_predictions=clip_predictions,
            clip_margin_scales=clip_margin_scales,
        )
    if learner_name == "random_forest":
        return build_random_forest_fit_predict(
            model_cfg,
            params=params,
            random_state=random_state,
            clip_predictions=clip_predictions,
            clip_margin_scales=clip_margin_scales,
        )
    raise ValueError(f"unsupported strict unified tabular learner: {learner!r}")


def _build_tabular_fit_predict(
    cfg: UnifiedTabularHeadConfig,
    estimator_factory: EstimatorFactory,
) -> FitPredictFn:
    _validate_config(cfg)

    def fit_predict(fold: UnifiedCVFold) -> UnifiedCVFoldPrediction:
        features = _build_fold_features(fold, cfg.feature_config)
        y_train = _as_1d_float("y_train_sec", fold.y_train_sec)
        y_val = _as_1d_float("y_val_sec", fold.y_val_sec)
        if len(features.X_train) != len(y_train):
            raise ValueError(f"train row mismatch: X={len(features.X_train)}, y={len(y_train)}")
        if len(features.X_val) != len(y_val):
            raise ValueError(f"validation row mismatch: X={len(features.X_val)}, y={len(y_val)}")

        seed = int(cfg.random_state) + int(getattr(fold, "fold_id", 0))
        model = estimator_factory(cfg.params, seed)
        model.fit(features.X_train, y_train)

        raw_val = np.asarray(model.predict(features.X_val), dtype=np.float32).reshape(-1)
        raw_test = np.asarray(model.predict(features.X_test), dtype=np.float32).reshape(-1)
        clip_state = _select_clip_state(raw_val, y_val, y_train, cfg.clipping)
        val_pred = _guard_predictions(raw_val, y_train=y_train, clip_state=clip_state)
        test_pred = _guard_predictions(raw_test, y_train=y_train, clip_state=clip_state)

        return UnifiedCVFoldPrediction(
            fold_id=int(getattr(fold, "fold_id", 0)),
            val_pred_sec=val_pred,
            test_pred_sec=test_pred,
            val_meta=fold.val_meta,
            test_meta=fold.test_meta,
            model_info={
                "learner": cfg.learner,
                "n_model": 1,
                "n_final_models": 1,
                "n_models_per_dataset": 0,
                "n_pooled_model_fits": 1,
                "fit_scope": "pooled_train_only",
                "prediction_scope": "pooled_val_test",
                "selected_params": _plain_params(cfg.params),
                "global_params": {
                    "random_state": int(cfg.random_state),
                    "effective_random_state": seed,
                    "clip_predictions": bool(cfg.clipping.enabled),
                    "clip_margin_scales": [float(x) for x in cfg.clipping.margin_scales],
                },
                "clip": clip_state.model_info(),
                "feature_source": features.source,
                "n_features": int(features.n_features),
                "train_rows": int(len(y_train)),
                "validation_rows": int(len(y_val)),
                "test_rows": int(len(features.X_test)),
                "selection_uses": ["X_train_mol", "X_train_cp", "y_train_sec", "X_val_mol", "X_val_cp", "y_val_sec"],
                "prediction_uses": ["X_test_mol", "X_test_cp"],
                "selection_excludes": [
                    "dataset_id_feature",
                    "dataset_indices",
                    "test_slices",
                    "lookup",
                    "HILIC_pool",
                    "local_fast",
                    "per_dataset_overrides",
                ],
            },
        )

    return fit_predict


@dataclass(frozen=True)
class _ClipState:
    enabled: bool
    lower: float
    upper: float
    margin_scale: float | None
    train_min: float
    train_max: float
    fallback: float
    validation_mae_raw: float | None
    validation_mae_clipped: float | None

    def model_info(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "selected_margin_scale": None if self.margin_scale is None else float(self.margin_scale),
            "train_min": float(self.train_min),
            "train_max": float(self.train_max),
            "fallback": float(self.fallback),
            "validation_mae_raw": self.validation_mae_raw,
            "validation_mae_clipped": self.validation_mae_clipped,
        }


def _select_clip_state(
    val_pred: np.ndarray,
    y_val: np.ndarray,
    y_train: np.ndarray,
    clipping: PredictionClipConfig,
) -> _ClipState:
    finite_train = y_train[np.isfinite(y_train)]
    if len(finite_train) == 0:
        raise ValueError("strict unified tabular heads require finite pooled train targets")
    train_min = float(np.min(finite_train))
    train_max = float(np.max(finite_train))
    fallback = float(np.mean(finite_train))
    span = max(train_max - train_min, 1.0, abs(fallback))

    raw_guarded = _replace_nonfinite(val_pred, fallback)
    finite_y_val = np.asarray(y_val, dtype=np.float32)
    validation_mae_raw = _safe_mae(finite_y_val, raw_guarded)
    if not clipping.enabled:
        return _ClipState(
            enabled=False,
            lower=float("-inf"),
            upper=float("inf"),
            margin_scale=None,
            train_min=train_min,
            train_max=train_max,
            fallback=fallback,
            validation_mae_raw=validation_mae_raw,
            validation_mae_clipped=None,
        )

    best_key: tuple[float, int] | None = None
    best_scale = 0.0
    best_bounds = (train_min, train_max)
    best_mae = validation_mae_raw
    for index, scale in enumerate(clipping.margin_scales):
        margin = max(0.0, float(scale)) * span
        lower = train_min - margin
        upper = train_max + margin
        clipped = np.clip(raw_guarded, lower, upper)
        mae = _safe_mae(finite_y_val, clipped)
        key = (mae if mae is not None else float("inf"), index)
        if best_key is None or key < best_key:
            best_key = key
            best_scale = max(0.0, float(scale))
            best_bounds = (float(lower), float(upper))
            best_mae = mae

    return _ClipState(
        enabled=True,
        lower=best_bounds[0],
        upper=best_bounds[1],
        margin_scale=best_scale,
        train_min=train_min,
        train_max=train_max,
        fallback=fallback,
        validation_mae_raw=validation_mae_raw,
        validation_mae_clipped=best_mae,
    )


def _guard_predictions(pred: np.ndarray, *, y_train: np.ndarray, clip_state: _ClipState) -> np.ndarray:
    guarded = _replace_nonfinite(pred, clip_state.fallback)
    if clip_state.enabled:
        guarded = np.clip(guarded, clip_state.lower, clip_state.upper)
    if not np.all(np.isfinite(guarded)):
        finite_train = y_train[np.isfinite(y_train)]
        fallback = float(np.mean(finite_train)) if len(finite_train) else 0.0
        guarded = _replace_nonfinite(guarded, fallback)
    return np.asarray(guarded, dtype=np.float32).reshape(-1)


def _replace_nonfinite(values: np.ndarray, fallback: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.where(np.isfinite(arr), arr, np.float32(fallback)).astype(np.float32, copy=False)


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return None
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


def _build_fold_features(fold: UnifiedCVFold, feature_config: Any | None) -> UnifiedTabularFeatureBlock:
    try:
        from .unified_features import build_strict_unified_fold_features

        features = build_strict_unified_fold_features(fold, config=feature_config)
        X_train = _as_2d_float("features.X_train", features.X_train)
        X_val = _as_2d_float("features.X_val", features.X_val)
        X_test = _as_2d_float("features.X_test", features.X_test)
        return UnifiedTabularFeatureBlock(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            source="unified_features.build_strict_unified_fold_features",
            n_features=int(X_train.shape[1]),
        )
    except (ImportError, AttributeError, TypeError):
        X_train = _as_2d_float("fold.X_train", fold.X_train)
        X_val = _as_2d_float("fold.X_val", fold.X_val)
        X_test = _as_2d_float("fold.X_test", fold.X_test)
        return UnifiedTabularFeatureBlock(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            source="fold.X",
            n_features=int(X_train.shape[1]),
        )


def _head_config(
    learner: LearnerName,
    model_cfg: Mapping[str, Any] | None,
    *,
    params: Mapping[str, Any] | None,
    random_state: int,
    clip_predictions: bool | None,
    clip_margin_scales: Sequence[float] | None,
) -> UnifiedTabularHeadConfig:
    cfg = dict(model_cfg or {})
    learner_params = _params_for_learner(learner, cfg)
    if params is not None:
        learner_params.update(copy.deepcopy(dict(params)))

    feature_config = cfg.get("UNIFIED_TABULAR_FEATURE_CONFIG", cfg.get("STRICT_UNIFIED_FEATURE_CONFIG"))
    clip_enabled = bool(
        _lookup_first(
            cfg,
            "UNIFIED_TABULAR_CLIP_PREDICTIONS",
            "STRICT_UNIFIED_CLIP_PREDICTIONS",
            default=False,
        )
        if clip_predictions is None
        else clip_predictions
    )
    margins = (
        clip_margin_scales
        if clip_margin_scales is not None
        else _lookup_first(
            cfg,
            "UNIFIED_TABULAR_CLIP_MARGIN_SCALES",
            "STRICT_UNIFIED_CLIP_MARGIN_SCALES",
            default=(0.0, 0.25, 0.5, 1.0),
        )
    )
    return UnifiedTabularHeadConfig(
        learner=learner,
        params=learner_params,
        random_state=int(_lookup_first(cfg, "UNIFIED_TABULAR_RANDOM_STATE", default=random_state)),
        feature_config=feature_config,
        clipping=PredictionClipConfig(enabled=clip_enabled, margin_scales=_float_tuple(margins)),
    )


def _params_for_learner(learner: str, model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    defaults = {
        "hist_gradient_boosting": {"max_iter": 200, "learning_rate": 0.05, "l2_regularization": 0.0},
        "extra_trees": {"n_estimators": 256, "min_samples_leaf": 2, "n_jobs": -1},
        "random_forest": {"n_estimators": 256, "min_samples_leaf": 2, "n_jobs": -1},
    }[learner]
    key_map = {
        "hist_gradient_boosting": ("UNIFIED_TABULAR_HGBR", "STRICT_UNIFIED_HGBR", "HIST_GRADIENT_BOOSTING"),
        "extra_trees": ("UNIFIED_TABULAR_EXTRA_TREES", "STRICT_UNIFIED_EXTRA_TREES", "EXTRA_TREES"),
        "random_forest": ("UNIFIED_TABULAR_RANDOM_FOREST", "STRICT_UNIFIED_RANDOM_FOREST", "RANDOM_FOREST"),
    }
    params = copy.deepcopy(defaults)
    for key in key_map[learner]:
        raw = model_cfg.get(key)
        if isinstance(raw, Mapping):
            params.update(copy.deepcopy(dict(raw)))
    return params


def _hist_gradient_boosting_factory(params: Mapping[str, Any], random_state: int) -> HistGradientBoostingRegressor:
    clean = dict(params)
    clean.setdefault("random_state", int(random_state))
    return HistGradientBoostingRegressor(**clean)


def _extra_trees_factory(params: Mapping[str, Any], random_state: int) -> ExtraTreesRegressor:
    clean = dict(params)
    clean.setdefault("random_state", int(random_state))
    return ExtraTreesRegressor(**clean)


def _random_forest_factory(params: Mapping[str, Any], random_state: int) -> RandomForestRegressor:
    clean = dict(params)
    clean.setdefault("random_state", int(random_state))
    return RandomForestRegressor(**clean)


def _validate_config(cfg: UnifiedTabularHeadConfig) -> None:
    _reject_forbidden_param_keys(cfg.params, path=cfg.learner)
    if not cfg.clipping.margin_scales:
        raise ValueError("clip margin scale grid must not be empty")
    if any(float(x) < 0 for x in cfg.clipping.margin_scales):
        raise ValueError("clip margin scales must be non-negative")


def _reject_forbidden_param_keys(value: Any, *, path: str) -> None:
    forbidden = {
        "dataset_id",
        "dataset_ids",
        "dataset_feature",
        "dataset_features",
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
                raise ValueError(f"{path}.{key} violates strict unified nModel=1 constraints")
            _reject_forbidden_param_keys(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, nested in enumerate(value):
            _reject_forbidden_param_keys(nested, path=f"{path}[{i}]")


def _normalise_learner_name(value: Any) -> LearnerName:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "hgb": "hist_gradient_boosting",
        "hgbr": "hist_gradient_boosting",
        "histgradientboosting": "hist_gradient_boosting",
        "hist_gradient_boosting_regressor": "hist_gradient_boosting",
        "extratrees": "extra_trees",
        "extra_trees_regressor": "extra_trees",
        "rf": "random_forest",
        "randomforest": "random_forest",
        "random_forest_regressor": "random_forest",
    }
    out = aliases.get(text, text)
    if out not in {"hist_gradient_boosting", "extra_trees", "random_forest"}:
        raise ValueError(f"unsupported strict unified tabular learner: {value!r}")
    return out  # type: ignore[return-value]


def _lookup_first(mapping: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    if not mapping:
        return default
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _float_tuple(values: Any) -> tuple[float, ...]:
    if isinstance(values, str):
        values = [item.strip() for item in values.split(",") if item.strip()]
    return tuple(float(x) for x in values)


def _as_1d_float(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return arr


def _as_2d_float(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        finite_cols = np.isfinite(arr)
        counts = finite_cols.sum(axis=0)
        sums = np.where(finite_cols, arr, 0.0).sum(axis=0)
        means = np.divide(sums, counts, out=np.zeros(arr.shape[1], dtype=np.float32), where=counts > 0)
        arr = np.where(np.isfinite(arr), arr, means.reshape(1, -1)).astype(np.float32)
    return arr.astype(np.float32, copy=False)


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


__all__ = [
    "PredictionClipConfig",
    "UnifiedTabularFeatureBlock",
    "UnifiedTabularHeadConfig",
    "build_extra_trees_fit_predict",
    "build_hist_gradient_boosting_fit_predict",
    "build_random_forest_fit_predict",
    "build_unified_tabular_fit_predict",
]

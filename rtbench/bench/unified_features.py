from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StrictUnifiedFeatureConfig:
    """Feature construction knobs for one pooled nModel=1 learner.

    All fitted state is learned from the pooled train split only. The default
    feature map is deterministic and does not accept dataset ids, targets, or
    split labels.
    """

    cp_interaction_components: int = 8
    include_cp_summary: bool = True
    include_interactions: bool = True
    include_condition_residuals: bool = True
    ridge: float = 1e-6
    eps: float = 1e-6
    enable_robust_preprocessing: bool = False
    robust_mol: bool = True
    robust_cp: bool = True
    robust_features: bool = True
    robust_clip: bool = True
    robust_scale: bool = True
    robust_quantile_low: float = 0.01
    robust_quantile_high: float = 0.99


@dataclass(frozen=True)
class StrictUnifiedFeatureSet:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    encoder: "StrictUnifiedFeatureEncoder"
    feature_names: tuple[str, ...]


class StrictUnifiedFeatureEncoder:
    """Train-only mol/CP encoder for strict unified CV folds.

    The encoder fits pooled-train imputers/scalers plus a CP-conditioned
    residualizer for molecule descriptors. Transforming validation/test rows
    reuses that state without looking at held-out rows or targets.
    """

    def __init__(self, config: StrictUnifiedFeatureConfig | None = None) -> None:
        self.config = _coerce_feature_config(config)
        self._fitted = False

    def fit(self, X_mol_train: Any, X_cp_train: Any) -> "StrictUnifiedFeatureEncoder":
        mol = _as_2d_float("X_mol_train", X_mol_train)
        cp = _as_2d_float("X_cp_train", X_cp_train)
        if mol.shape[0] != cp.shape[0]:
            raise ValueError(f"train row mismatch: X_mol has {mol.shape[0]} rows, X_cp has {cp.shape[0]} rows")
        if mol.shape[0] == 0:
            raise ValueError("strict unified features require at least one pooled train row")

        self.mol_robust_preprocessor_ = self._make_robust_preprocessor(enabled=self.config.robust_mol)
        self.cp_robust_preprocessor_ = self._make_robust_preprocessor(enabled=self.config.robust_cp)
        mol_base = self.mol_robust_preprocessor_.fit_transform(mol)
        cp_base = self.cp_robust_preprocessor_.fit_transform(cp)
        self.mol_robust_lower_ = self.mol_robust_preprocessor_.lower_
        self.mol_robust_upper_ = self.mol_robust_preprocessor_.upper_
        self.cp_robust_lower_ = self.cp_robust_preprocessor_.lower_
        self.cp_robust_upper_ = self.cp_robust_preprocessor_.upper_

        self.mol_mean_, self.mol_scale_ = _fit_center_scale(mol_base, eps=self.config.eps)
        self.cp_mean_, self.cp_scale_ = _fit_center_scale(cp_base, eps=self.config.eps)

        cp_summary = _cp_summary(cp_base)
        self.cp_summary_mean_, self.cp_summary_scale_ = _fit_center_scale(cp_summary, eps=self.config.eps)

        mol_z = _transform_center_scale(mol_base, self.mol_mean_, self.mol_scale_)
        cp_z = _transform_center_scale(cp_base, self.cp_mean_, self.cp_scale_)
        summary_z = _transform_center_scale(cp_summary, self.cp_summary_mean_, self.cp_summary_scale_)
        basis = self._condition_basis(cp_z, summary_z)
        self.condition_coef_ = _fit_ridge_residualizer(basis, mol_z, ridge=self.config.ridge)

        self.n_mol_features_ = int(mol.shape[1])
        self.n_cp_features_ = int(cp.shape[1])
        self.n_cp_interaction_terms_ = int(self._interaction_terms(cp_z, summary_z).shape[1])
        self.feature_names_ = self._make_feature_names()
        self.n_features_out_ = len(self.feature_names_)
        train_out = self._assemble_features(mol_z, cp_z, summary_z)
        self.feature_robust_preprocessor_ = self._make_robust_preprocessor(enabled=self.config.robust_features)
        self.feature_robust_preprocessor_.fit(train_out)
        self.feature_robust_lower_ = self.feature_robust_preprocessor_.lower_
        self.feature_robust_upper_ = self.feature_robust_preprocessor_.upper_
        self._fitted = True
        return self

    def transform(self, X_mol: Any, X_cp: Any) -> np.ndarray:
        self._require_fitted()
        mol = _as_2d_float("X_mol", X_mol)
        cp = _as_2d_float("X_cp", X_cp)
        if mol.shape[0] != cp.shape[0]:
            raise ValueError(f"row mismatch: X_mol has {mol.shape[0]} rows, X_cp has {cp.shape[0]} rows")
        if mol.shape[1] != self.n_mol_features_:
            raise ValueError(f"molecule feature dimension changed: got={mol.shape[1]}, expected={self.n_mol_features_}")
        if cp.shape[1] != self.n_cp_features_:
            raise ValueError(f"CP feature dimension changed: got={cp.shape[1]}, expected={self.n_cp_features_}")

        mol_base = self.mol_robust_preprocessor_.transform(mol)
        cp_base = self.cp_robust_preprocessor_.transform(cp)
        mol_z = _transform_center_scale(mol_base, self.mol_mean_, self.mol_scale_)
        cp_z = _transform_center_scale(cp_base, self.cp_mean_, self.cp_scale_)
        summary_z = _transform_center_scale(_cp_summary(cp_base), self.cp_summary_mean_, self.cp_summary_scale_)

        out = self.feature_robust_preprocessor_.transform(self._assemble_features(mol_z, cp_z, summary_z))
        if out.shape[1] != self.n_features_out_:
            raise AssertionError(
                f"strict unified feature dimension mismatch: got={out.shape[1]}, expected={self.n_features_out_}"
            )
        return _finite_float32(out)

    def _assemble_features(self, mol_z: np.ndarray, cp_z: np.ndarray, summary_z: np.ndarray) -> np.ndarray:
        parts = [mol_z, cp_z]
        if self.config.include_cp_summary:
            parts.append(summary_z)
        if self.config.include_interactions:
            parts.append(_pairwise_interactions(mol_z, self._interaction_terms(cp_z, summary_z)))
        if self.config.include_condition_residuals:
            basis = self._condition_basis(cp_z, summary_z)
            parts.append((mol_z - basis @ self.condition_coef_).astype(np.float32))
        out = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        return _finite_float32(out)

    def fit_transform(self, X_mol_train: Any, X_cp_train: Any) -> np.ndarray:
        self.fit(X_mol_train, X_cp_train)
        return self.transform(X_mol_train, X_cp_train)

    def _make_robust_preprocessor(self, *, enabled: bool) -> "_TrainOnlyRobustPreprocessor":
        return _TrainOnlyRobustPreprocessor(
            enabled=bool(self.config.enable_robust_preprocessing and enabled),
            clip=bool(self.config.robust_clip),
            scale=bool(self.config.robust_scale),
            q_low=float(self.config.robust_quantile_low),
            q_high=float(self.config.robust_quantile_high),
            eps=float(self.config.eps),
        )

    def _condition_basis(self, cp_z: np.ndarray, summary_z: np.ndarray) -> np.ndarray:
        terms = self._interaction_terms(cp_z, summary_z)
        intercept = np.ones((cp_z.shape[0], 1), dtype=np.float32)
        return np.concatenate([intercept, terms], axis=1).astype(np.float32, copy=False)

    def _interaction_terms(self, cp_z: np.ndarray, summary_z: np.ndarray) -> np.ndarray:
        n_cp = min(max(0, int(self.config.cp_interaction_components)), cp_z.shape[1])
        pieces: list[np.ndarray] = []
        if n_cp:
            pieces.append(cp_z[:, :n_cp])
        if self.config.include_cp_summary:
            pieces.append(summary_z)
        if not pieces:
            return np.zeros((cp_z.shape[0], 0), dtype=np.float32)
        return np.concatenate(pieces, axis=1).astype(np.float32, copy=False)

    def _make_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        names.extend(f"mol_z[{i}]" for i in range(self.n_mol_features_))
        names.extend(f"cp_z[{i}]" for i in range(self.n_cp_features_))
        if self.config.include_cp_summary:
            names.extend(f"cp_summary_z[{name}]" for name in _CP_SUMMARY_NAMES)
        if self.config.include_interactions:
            for i in range(self.n_mol_features_):
                for j in range(self.n_cp_interaction_terms_):
                    names.append(f"mol_cp_interaction[{i},{j}]")
        if self.config.include_condition_residuals:
            names.extend(f"mol_condition_residual[{i}]" for i in range(self.n_mol_features_))
        return tuple(names)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("StrictUnifiedFeatureEncoder must be fitted before transform")


def build_strict_unified_features(
    *,
    X_mol_train: Any,
    X_cp_train: Any,
    X_mol_val: Any,
    X_cp_val: Any,
    X_mol_test: Any,
    X_cp_test: Any,
    config: StrictUnifiedFeatureConfig | None = None,
) -> StrictUnifiedFeatureSet:
    encoder = StrictUnifiedFeatureEncoder(config=config)
    encoder.fit(X_mol_train, X_cp_train)
    return StrictUnifiedFeatureSet(
        X_train=encoder.transform(X_mol_train, X_cp_train),
        X_val=encoder.transform(X_mol_val, X_cp_val),
        X_test=encoder.transform(X_mol_test, X_cp_test),
        encoder=encoder,
        feature_names=encoder.feature_names_,
    )


def build_strict_unified_fold_features(fold: Any, config: StrictUnifiedFeatureConfig | None = None) -> StrictUnifiedFeatureSet:
    """Build strict features from any object exposing UnifiedCVFold arrays."""

    return build_strict_unified_features(
        X_mol_train=fold.X_train_mol,
        X_cp_train=fold.X_train_cp,
        X_mol_val=fold.X_val_mol,
        X_cp_val=fold.X_val_cp,
        X_mol_test=fold.X_test_mol,
        X_cp_test=fold.X_test_cp,
        config=config,
    )


def fit_strict_unified_feature_encoder(
    X_mol_train: Any,
    X_cp_train: Any,
    config: StrictUnifiedFeatureConfig | None = None,
) -> StrictUnifiedFeatureEncoder:
    return StrictUnifiedFeatureEncoder(config=config).fit(X_mol_train, X_cp_train)


def _coerce_feature_config(config: StrictUnifiedFeatureConfig | Mapping[str, Any] | None) -> StrictUnifiedFeatureConfig:
    if config is None:
        return StrictUnifiedFeatureConfig()
    if isinstance(config, StrictUnifiedFeatureConfig):
        return config
    if isinstance(config, Mapping):
        return StrictUnifiedFeatureConfig(**dict(config))
    raise TypeError(f"unsupported strict unified feature config type: {type(config)!r}")


_CP_SUMMARY_NAMES = ("finite_fraction", "mean", "std", "min", "max", "l1_mean", "l2_norm", "max_abs")
_FLOAT32_MAX = float(np.finfo(np.float32).max)


class _TrainOnlyRobustPreprocessor:
    def __init__(
        self,
        *,
        enabled: bool,
        clip: bool,
        scale: bool,
        q_low: float,
        q_high: float,
        eps: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.clip = bool(clip)
        self.scale = bool(scale)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.eps = float(eps)
        self._fitted = False

    def fit(self, X: np.ndarray) -> "_TrainOnlyRobustPreprocessor":
        X64 = np.asarray(X, dtype=np.float64)
        if X64.ndim != 2:
            raise ValueError(f"robust preprocessor requires a 2D array, got shape={X64.shape}")
        n_features = int(X64.shape[1])
        self.lower_ = np.full(n_features, -_FLOAT32_MAX, dtype=np.float32)
        self.upper_ = np.full(n_features, _FLOAT32_MAX, dtype=np.float32)
        self.center_ = np.zeros(n_features, dtype=np.float32)
        self.scale_ = np.ones(n_features, dtype=np.float32)
        if not self.enabled:
            self._fitted = True
            return self
        if not (0.0 <= self.q_low < self.q_high <= 1.0):
            raise ValueError(
                f"robust quantiles must satisfy 0 <= low < high <= 1, got low={self.q_low}, high={self.q_high}"
            )

        for col in range(n_features):
            finite_values = X64[np.isfinite(X64[:, col]), col]
            if finite_values.size == 0:
                continue
            if self.clip:
                lower = float(np.quantile(finite_values, self.q_low))
                upper = float(np.quantile(finite_values, self.q_high))
                if not np.isfinite(lower):
                    lower = 0.0
                if not np.isfinite(upper):
                    upper = lower
                if upper < lower:
                    upper = lower
                self.lower_[col] = np.float32(np.clip(lower, -_FLOAT32_MAX, _FLOAT32_MAX))
                self.upper_[col] = np.float32(np.clip(upper, -_FLOAT32_MAX, _FLOAT32_MAX))
                finite_values = np.clip(finite_values, lower, upper)
            center = float(np.median(finite_values))
            if not np.isfinite(center):
                center = 0.0
            self.center_[col] = np.float32(np.clip(center, -_FLOAT32_MAX, _FLOAT32_MAX))
            if self.scale:
                q75 = float(np.quantile(finite_values, 0.75))
                q25 = float(np.quantile(finite_values, 0.25))
                scale = q75 - q25
                if not np.isfinite(scale) or scale <= self.eps:
                    scale = 1.0
                self.scale_[col] = np.float32(np.clip(scale, self.eps, _FLOAT32_MAX))
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("robust preprocessor must be fitted before transform")
        if not self.enabled:
            return np.asarray(X, dtype=np.float32)
        X64 = np.asarray(X, dtype=np.float64)
        filled = np.where(np.isfinite(X64), X64, self.center_.reshape(1, -1).astype(np.float64))
        if self.clip:
            filled = np.clip(filled, self.lower_.reshape(1, -1), self.upper_.reshape(1, -1))
        if self.scale:
            filled = (filled - self.center_.reshape(1, -1)) / self.scale_.reshape(1, -1)
        return _finite_float32(filled)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def _as_2d_float(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape={arr.shape}")
    return arr


def _fit_center_scale(X: np.ndarray, *, eps: float) -> tuple[np.ndarray, np.ndarray]:
    X64 = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X64)
    counts = finite.sum(axis=0)
    sums = np.where(finite, X64, 0.0).sum(axis=0)
    mean = np.divide(sums, counts, out=np.zeros(X64.shape[1], dtype=np.float64), where=counts > 0)
    centered = np.where(finite, X64 - mean, 0.0)
    var = np.divide((centered * centered).sum(axis=0), counts, out=np.zeros(X64.shape[1], dtype=np.float64), where=counts > 0)
    scale = np.sqrt(var)
    scale = np.where(np.isfinite(scale) & (scale > float(eps)), scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def _transform_center_scale(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    X64 = np.asarray(X, dtype=np.float64)
    mean64 = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    scale64 = np.asarray(scale, dtype=np.float64).reshape(1, -1)
    filled = np.where(np.isfinite(X64), X64, mean64)
    return _finite_float32((filled - mean64) / scale64)


def _cp_summary(cp: np.ndarray) -> np.ndarray:
    X = np.asarray(cp, dtype=np.float64)
    n_rows, n_cols = X.shape
    out = np.zeros((n_rows, len(_CP_SUMMARY_NAMES)), dtype=np.float64)
    if n_cols == 0:
        return out

    finite = np.isfinite(X)
    safe = np.where(finite, X, 0.0)
    counts = finite.sum(axis=1).astype(np.float64)
    has = counts > 0
    out[:, 0] = counts / float(n_cols)
    if not np.any(has):
        return out

    sums = safe.sum(axis=1)
    mean = np.divide(sums, counts, out=np.zeros(n_rows, dtype=np.float64), where=has)
    centered = np.where(finite, X - mean.reshape(-1, 1), 0.0)
    var = np.divide((centered * centered).sum(axis=1), counts, out=np.zeros(n_rows, dtype=np.float64), where=has)
    masked_pos = np.where(finite, X, np.inf)
    masked_neg = np.where(finite, X, -np.inf)
    min_v = np.min(masked_pos, axis=1)
    max_v = np.max(masked_neg, axis=1)
    abs_safe = np.abs(safe)

    out[:, 1] = mean
    out[:, 2] = np.sqrt(np.maximum(var, 0.0))
    out[:, 3] = np.where(has, min_v, 0.0)
    out[:, 4] = np.where(has, max_v, 0.0)
    out[:, 5] = np.divide(abs_safe.sum(axis=1), counts, out=np.zeros(n_rows, dtype=np.float64), where=has)
    out[:, 6] = np.sqrt((safe * safe).sum(axis=1))
    out[:, 7] = np.max(abs_safe, axis=1)
    return _finite_float32(out)


def _fit_ridge_residualizer(basis: np.ndarray, target: np.ndarray, *, ridge: float) -> np.ndarray:
    B = np.asarray(basis, dtype=np.float64)
    Y = np.asarray(target, dtype=np.float64)
    penalty = max(0.0, float(ridge)) * np.eye(B.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(B.T @ B + penalty) @ B.T @ Y
    return coef.astype(np.float32)


def _pairwise_interactions(mol_z: np.ndarray, terms: np.ndarray) -> np.ndarray:
    if terms.shape[1] == 0:
        return np.zeros((mol_z.shape[0], 0), dtype=np.float32)
    return _finite_float32(np.asarray(mol_z, dtype=np.float64)[:, :, None] * np.asarray(terms, dtype=np.float64)[:, None, :]).reshape(
        mol_z.shape[0], -1
    )


def _finite_float32(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=_FLOAT32_MAX, neginf=-_FLOAT32_MAX)
    arr = np.clip(arr, -_FLOAT32_MAX, _FLOAT32_MAX)
    return arr.astype(np.float32, copy=False)


__all__ = [
    "StrictUnifiedFeatureConfig",
    "StrictUnifiedFeatureEncoder",
    "StrictUnifiedFeatureSet",
    "build_strict_unified_features",
    "build_strict_unified_fold_features",
    "fit_strict_unified_feature_encoder",
]

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Config, resolve_config
from ..metrics import compute_metrics
from ..models import kfold_split
from .prepare import config_from_raw, prepare, raw_from_config


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class UnifiedCVRowMeta:
    dataset_id: str
    original_row_id: str
    local_row_index: int
    split_name: SplitName
    fold_id: int


@dataclass(frozen=True)
class DatasetFoldIndices:
    dataset_id: str
    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class UnifiedCVFold:
    """Pooled strict nModel=1 train/validation/test arrays for one CV fold.

    A downstream learner should fit exactly one model using the pooled train rows
    and may use the pooled validation rows for early stopping/model selection.
    It should then predict `X_val` and `X_test` in one call. Per-dataset logic is
    intentionally absent; `test_slices` only maps the pooled test predictions
    back to dataset ids for reporting.
    """

    fold_id: int
    dataset_ids: tuple[str, ...]
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    X_train_mol: np.ndarray
    X_val_mol: np.ndarray
    X_test_mol: np.ndarray
    X_train_cp: np.ndarray
    X_val_cp: np.ndarray
    X_test_cp: np.ndarray
    y_train_sec: np.ndarray
    y_val_sec: np.ndarray
    y_test_sec: np.ndarray
    train_meta: tuple[UnifiedCVRowMeta, ...]
    val_meta: tuple[UnifiedCVRowMeta, ...]
    test_meta: tuple[UnifiedCVRowMeta, ...]
    dataset_indices: Mapping[str, DatasetFoldIndices]
    test_slices: Mapping[str, slice]


@dataclass(frozen=True)
class UnifiedCVFoldPrediction:
    fold_id: int
    val_pred_sec: np.ndarray
    test_pred_sec: np.ndarray
    val_meta: tuple[UnifiedCVRowMeta, ...]
    test_meta: tuple[UnifiedCVRowMeta, ...]
    model_info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnifiedCVRunResult:
    out_root: Path
    per_seed_df: pd.DataFrame
    per_seed_path: Path
    predictions_path: Path
    fold_predictions: tuple[UnifiedCVFoldPrediction, ...]
    sheet: str | None = None
    mode: str | None = None
    n_model: int = 1


FitPredictFn = Callable[
    [UnifiedCVFold],
    UnifiedCVFoldPrediction | tuple[np.ndarray, np.ndarray] | Mapping[str, Any],
]


def normalize_dataset_ids(dataset_ids: Iterable[str]) -> tuple[str, ...]:
    raw_ids = tuple(str(dataset_id).strip() for dataset_id in dataset_ids)
    if not raw_ids:
        raise ValueError("unified CV requires at least one dataset id")
    if any(not dataset_id for dataset_id in raw_ids):
        raise ValueError("dataset ids must be non-empty")
    ids = tuple(dataset_id.zfill(4) for dataset_id in raw_ids)
    if len(set(ids)) != len(ids):
        raise ValueError(f"dataset ids must be unique after normalization: {ids}")
    return ids


def build_dataset_fold_indices(
    mats: Mapping[str, Any],
    dataset_ids: Iterable[str],
    fold_id: int,
    *,
    n_splits: int = 10,
    shuffle_seed: int = 0,
    val_fold_offset: int = 1,
) -> dict[str, DatasetFoldIndices]:
    """Return per-dataset indices using existing `kfold_split` semantics."""

    ids = normalize_dataset_ids(dataset_ids)
    out: dict[str, DatasetFoldIndices] = {}
    for dataset_id in ids:
        if dataset_id not in mats:
            raise KeyError(f"dataset {dataset_id} is missing from mats")
        mat = mats[dataset_id]
        split = kfold_split(
            y=np.asarray(mat.y_sec),
            seed=int(fold_id),
            n_splits=int(n_splits),
            shuffle_seed=int(shuffle_seed),
            val_fold_offset=int(val_fold_offset),
        )
        out[dataset_id] = DatasetFoldIndices(
            dataset_id=dataset_id,
            fold_id=int(fold_id) % int(n_splits),
            train_idx=np.asarray(split.train_idx, dtype=int),
            val_idx=np.asarray(split.val_idx, dtype=int),
            test_idx=np.asarray(split.test_idx, dtype=int),
        )
    return out


def assemble_unified_cv_fold(
    mats: Mapping[str, Any],
    dataset_ids: Iterable[str],
    fold_id: int,
    *,
    n_splits: int = 10,
    shuffle_seed: int = 0,
    val_fold_offset: int = 1,
) -> UnifiedCVFold:
    """Pool all datasets into one train/validation/test fold.

    For every dataset, the test fold is `fold_id % n_splits`, validation is the
    adjacent fold selected by `val_fold_offset`, and train is the remaining
    folds. This delegates split construction to `rtbench.models.kfold_split`.
    """

    ids = normalize_dataset_ids(dataset_ids)
    dataset_indices = build_dataset_fold_indices(
        mats,
        ids,
        int(fold_id),
        n_splits=int(n_splits),
        shuffle_seed=int(shuffle_seed),
        val_fold_offset=int(val_fold_offset),
    )

    train_parts = _collect_split_parts(mats, ids, dataset_indices, "train")
    val_parts = _collect_split_parts(mats, ids, dataset_indices, "val")
    test_parts = _collect_split_parts(mats, ids, dataset_indices, "test")

    test_slices: dict[str, slice] = {}
    cursor = 0
    for dataset_id in ids:
        n_rows = len(dataset_indices[dataset_id].test_idx)
        test_slices[dataset_id] = slice(cursor, cursor + n_rows)
        cursor += n_rows

    return UnifiedCVFold(
        fold_id=int(fold_id) % int(n_splits),
        dataset_ids=ids,
        X_train=_concat_arrays(train_parts["X"]),
        X_val=_concat_arrays(val_parts["X"]),
        X_test=_concat_arrays(test_parts["X"]),
        X_train_mol=_concat_arrays(train_parts["X_mol"]),
        X_val_mol=_concat_arrays(val_parts["X_mol"]),
        X_test_mol=_concat_arrays(test_parts["X_mol"]),
        X_train_cp=_concat_arrays(train_parts["X_cp"]),
        X_val_cp=_concat_arrays(val_parts["X_cp"]),
        X_test_cp=_concat_arrays(test_parts["X_cp"]),
        y_train_sec=_concat_arrays(train_parts["y_sec"]).reshape(-1),
        y_val_sec=_concat_arrays(val_parts["y_sec"]).reshape(-1),
        y_test_sec=_concat_arrays(test_parts["y_sec"]).reshape(-1),
        train_meta=tuple(train_parts["meta"]),
        val_meta=tuple(val_parts["meta"]),
        test_meta=tuple(test_parts["meta"]),
        dataset_indices=dataset_indices,
        test_slices=test_slices,
    )


def iter_unified_cv_folds(
    mats: Mapping[str, Any],
    dataset_ids: Iterable[str],
    *,
    n_splits: int = 10,
    shuffle_seed: int = 0,
    val_fold_offset: int = 1,
) -> Iterable[UnifiedCVFold]:
    for fold_id in range(int(n_splits)):
        yield assemble_unified_cv_fold(
            mats,
            dataset_ids,
            fold_id,
            n_splits=int(n_splits),
            shuffle_seed=int(shuffle_seed),
            val_fold_offset=int(val_fold_offset),
        )


def run_strict_unified_cv_fold(fold: UnifiedCVFold, fit_predict: FitPredictFn) -> UnifiedCVFoldPrediction:
    """Run one strict nModel=1 fit/predict callback for a pooled fold."""

    raw = fit_predict(fold)
    pred = _coerce_fold_prediction(fold, raw)
    _validate_prediction_lengths(fold, pred)
    return pred


def evaluate_strict_unified_cv(
    mats: Mapping[str, Any],
    dataset_ids: Iterable[str],
    fit_predict: FitPredictFn,
    *,
    n_splits: int = 10,
    shuffle_seed: int = 0,
    val_fold_offset: int = 1,
) -> list[UnifiedCVFoldPrediction]:
    """Evaluate all folds with one pooled model fit per fold."""

    predictions: list[UnifiedCVFoldPrediction] = []
    for fold in iter_unified_cv_folds(
        mats,
        dataset_ids,
        n_splits=int(n_splits),
        shuffle_seed=int(shuffle_seed),
        val_fold_offset=int(val_fold_offset),
    ):
        predictions.append(run_strict_unified_cv_fold(fold, fit_predict))
    return predictions


def run_unified_cv(
    *,
    sheet: str | None = None,
    mode: str | None = None,
    config_path: str | Path | None = None,
    config: Config | Mapping[str, Any] | None = None,
    folds: int | None = None,
    n_folds: int | None = None,
    shuffle_seed: int | None = None,
    output_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    no_download: bool | None = None,
    download: bool | None = None,
    prepared: Any | None = None,
    fit_predict: FitPredictFn | None = None,
) -> UnifiedCVRunResult:
    """CLI-ready strict unified CV runner.

    This is intentionally minimal: it prepares configured external datasets,
    assembles pooled k-fold splits, fits one pooled model per fold, predicts all
    held-out test rows, and writes a per-fold metrics CSV compatible with
    `write_unirt_report`. The built-in learner is selected from the config:
    `UNIFIED_CV_LEARNER=hypertl` uses the strict pooled HyperTL learner, while
    unspecified configs keep the legacy `StandardScaler + Ridge` fallback.
    Callers can also inject a strict pooled learner via `fit_predict`.
    """

    repo_base = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    cfg, raw_cfg, resolved_config_path = _resolve_config_input(config_path=config_path, config=config, repo_base=repo_base)
    if output_root is None:
        output_root = cfg.outputs.get("root", "outputs_unified_cv")
    out_root = _resolve_output_root(output_root, repo_base=repo_base)

    if prepared is None:
        no_download_flag = _resolve_no_download(no_download=no_download, download=download)
        prepared = prepare(cfg, no_download=no_download_flag)

    dataset_ids = normalize_dataset_ids(getattr(prepared, "external_ids", cfg.datasets.get("external", ())))
    split_cfg = cfg.split or {}
    n_splits = int(
        folds
        if folds is not None
        else (n_folds if n_folds is not None else split_cfg.get("folds", split_cfg.get("n_splits", 10)))
    )
    fold_shuffle_seed = int(
        shuffle_seed
        if shuffle_seed is not None
        else split_cfg.get("shuffle_seed", split_cfg.get("seed", 0))
    )
    val_fold_offset = int(split_cfg.get("val_fold_offset", 1))
    learner = fit_predict if fit_predict is not None else _default_fit_predict(cfg.models)

    folds = list(
        iter_unified_cv_folds(
            prepared.mats,
            dataset_ids,
            n_splits=n_splits,
            shuffle_seed=fold_shuffle_seed,
            val_fold_offset=val_fold_offset,
        )
    )
    predictions = [run_strict_unified_cv_fold(fold, learner) for fold in folds]

    predictions_df = _prediction_frame_with_truth(prepared.mats, predictions)
    per_seed_df = _per_seed_metrics_from_predictions(predictions_df)

    metrics_root = out_root / "metrics"
    pred_root = out_root / "predictions"
    metrics_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    per_seed_path = metrics_root / "per_seed.csv"
    predictions_path = pred_root / "unified_cv_predictions.csv"
    per_seed_df.to_csv(per_seed_path, index=False, encoding="utf-8")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8")
    _write_optional_audit(out_root=out_root, folds=folds, predictions=predictions)

    _write_run_metadata(
        out_root=out_root,
        sheet=sheet,
        mode=mode,
        config_path=resolved_config_path,
        config_raw=raw_cfg,
        n_splits=n_splits,
        shuffle_seed=fold_shuffle_seed,
        dataset_ids=dataset_ids,
    )

    return UnifiedCVRunResult(
        out_root=out_root,
        per_seed_df=per_seed_df,
        per_seed_path=per_seed_path,
        predictions_path=predictions_path,
        fold_predictions=tuple(predictions),
        sheet=sheet,
        mode=mode,
        n_model=1,
    )


run_strict_unified_cv = run_unified_cv
run = run_unified_cv


def prediction_frame(predictions: Sequence[UnifiedCVFoldPrediction], *, split_name: SplitName = "test") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        if split_name == "test":
            meta = pred.test_meta
            values = np.asarray(pred.test_pred_sec, dtype=np.float32).reshape(-1)
        elif split_name == "val":
            meta = pred.val_meta
            values = np.asarray(pred.val_pred_sec, dtype=np.float32).reshape(-1)
        else:
            raise ValueError("prediction_frame supports split_name='test' or split_name='val'")
        if len(meta) != len(values):
            raise ValueError(f"prediction length mismatch for fold {pred.fold_id}: {len(values)} predictions, {len(meta)} rows")
        for row_meta, y_pred in zip(meta, values, strict=True):
            rows.append(
                {
                    "dataset": row_meta.dataset_id,
                    "fold": int(row_meta.fold_id),
                    "split": row_meta.split_name,
                    "original_row_id": row_meta.original_row_id,
                    "local_row_index": int(row_meta.local_row_index),
                    "y_pred_sec": float(y_pred),
                }
            )
    return pd.DataFrame(rows)


def hyper_tl_validation_index_kwargs(fold: UnifiedCVFold) -> dict[str, np.ndarray]:
    """Compatibility payload for future explicit HyperTL validation support.

    Expected downstream API shape:
    `pretrain_hyper_tl(..., train_idx=<pooled-train indices>,
    val_idx=<pooled-validation indices>)`, where the arrays index the
    concatenated `[fold.X_train; fold.X_val]` pool. `rtbench.hyper` does not
    currently accept these kwargs; callers should only pass them after checking
    that their selected compatibility wrapper supports the API.
    """

    n_train = len(fold.y_train_sec)
    n_val = len(fold.y_val_sec)
    return {
        "train_idx": np.arange(n_train, dtype=int),
        "val_idx": np.arange(n_train, n_train + n_val, dtype=int),
    }


def _resolve_config_input(
    *,
    config_path: str | Path | None,
    config: Config | Mapping[str, Any] | None,
    repo_base: Path,
) -> tuple[Config, dict[str, Any], Path | None]:
    if config is not None:
        if isinstance(config, Config):
            return config, raw_from_config(config), None
        if isinstance(config, (str, Path)):
            return _resolve_config_input(config_path=config, config=None, repo_base=repo_base)
        raw = copy.deepcopy(dict(config))
        return config_from_raw(raw), raw, None
    if config_path is None:
        raise ValueError("run_unified_cv requires either config_path or config")
    path = Path(config_path)
    if not path.is_absolute():
        path = repo_base / path
    resolved = resolve_config(path)
    return resolved.config, resolved.raw, resolved.path


def _resolve_output_root(output_root: str | Path, *, repo_base: Path) -> Path:
    out_root = Path(output_root)
    if not out_root.is_absolute():
        out_root = repo_base / out_root
    return out_root.resolve()


def _resolve_no_download(*, no_download: bool | None, download: bool | None) -> bool:
    if no_download is not None:
        return bool(no_download)
    if download is not None:
        return not bool(download)
    return False


def _default_fit_predict(model_cfg: Mapping[str, Any]) -> FitPredictFn:
    learner = str(
        model_cfg.get(
            "UNIFIED_CV_LEARNER",
            model_cfg.get("STRICT_UNIFIED_CV_LEARNER", ""),
        )
    ).strip().lower()
    wants_hypertl = learner in {"hypertl", "hyper_tl", "hyper-tl", "unified_hypertl", "unified-hypertl"}
    wants_tabular = learner in {
        "tabular",
        "unified_tabular",
        "unified-tabular",
        "hist_gradient_boosting",
        "hist-gradient-boosting",
        "hgbr",
        "extra_trees",
        "extra-trees",
        "random_forest",
        "random-forest",
    }
    wants_ridge = learner in {"", "ridge", "ridge_fallback", "ridge-fallback"}
    if not wants_hypertl and not wants_tabular and not wants_ridge:
        raise ValueError(f"unsupported strict unified CV learner: {learner!r}")

    if wants_hypertl or (learner == "" and _model_cfg_requests_hypertl(model_cfg)):
        from .unified_hypertl import build_unified_hypertl_fit_predict

        return build_unified_hypertl_fit_predict(model_cfg)
    if wants_tabular:
        from .unified_tabular import build_unified_tabular_fit_predict

        tabular_learner = None if learner in {"tabular", "unified_tabular", "unified-tabular"} else learner
        return build_unified_tabular_fit_predict(model_cfg, learner=tabular_learner)
    return _default_ridge_fit_predict(model_cfg)


def _model_cfg_requests_hypertl(model_cfg: Mapping[str, Any]) -> bool:
    return bool(
        model_cfg.get("ENABLE_UNIFIED_CV_HYPER_TL", False)
        or model_cfg.get("ENABLE_STRICT_UNIFIED_HYPER_TL", False)
        or (
            model_cfg.get("ONLY_HYPER_TL", False)
            and (model_cfg.get("ENABLE_HYPER_TL", False) or model_cfg.get("ENABLE_SHEET_UNIFIED_HYPER_TL", False))
        )
    )


def _default_ridge_fit_predict(model_cfg: Mapping[str, Any]) -> FitPredictFn:
    alpha = float(model_cfg.get("UNIFIED_CV_RIDGE_ALPHA", model_cfg.get("RIDGE_ALPHA", 1.0)))

    def fit_predict(fold: UnifiedCVFold) -> tuple[np.ndarray, np.ndarray]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(fold.X_train, fold.y_train_sec)
        return (
            model.predict(fold.X_val).astype(np.float32),
            model.predict(fold.X_test).astype(np.float32),
        )

    return fit_predict


def _prediction_frame_with_truth(
    mats: Mapping[str, Any],
    predictions: Sequence[UnifiedCVFoldPrediction],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        values = np.asarray(pred.test_pred_sec, dtype=np.float32).reshape(-1)
        if len(values) != len(pred.test_meta):
            raise ValueError(f"test prediction length mismatch for fold {pred.fold_id}")
        for row_meta, y_pred in zip(pred.test_meta, values, strict=True):
            mat = mats[row_meta.dataset_id]
            y_true = float(np.asarray(mat.y_sec, dtype=np.float32)[int(row_meta.local_row_index)])
            rows.append(
                {
                    "dataset": row_meta.dataset_id,
                    "seed": int(row_meta.fold_id),
                    "fold": int(row_meta.fold_id),
                    "split": row_meta.split_name,
                    "original_row_id": row_meta.original_row_id,
                    "local_row_index": int(row_meta.local_row_index),
                    "y_true_sec": y_true,
                    "y_pred_sec": float(y_pred),
                }
            )
    return pd.DataFrame(rows)


def _per_seed_metrics_from_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions_df.empty:
        return pd.DataFrame(columns=["dataset", "seed", "mae", "medae", "mre", "medre", "r2", "rmse"])
    grouped = predictions_df.groupby(["dataset", "seed"], sort=True)
    for (dataset_id, seed), cur in grouped:
        metrics = compute_metrics(
            cur["y_true_sec"].to_numpy(dtype=np.float64),
            cur["y_pred_sec"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "dataset": str(dataset_id).zfill(4),
                "seed": int(seed),
                "mae": metrics["mae"],
                "medae": metrics["medae"],
                "mre": metrics["mre"],
                "medre": metrics["medre"],
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "seed"]).reset_index(drop=True)


def _write_run_metadata(
    *,
    out_root: Path,
    sheet: str | None,
    mode: str | None,
    config_path: Path | None,
    config_raw: Mapping[str, Any],
    n_splits: int,
    shuffle_seed: int,
    dataset_ids: tuple[str, ...],
) -> None:
    metadata = {
        "sheet": sheet,
        "mode": mode,
        "config_path": None if config_path is None else str(config_path),
        "n_model": 1,
        "n_splits": int(n_splits),
        "shuffle_seed": int(shuffle_seed),
        "dataset_ids": list(dataset_ids),
        "metrics": dict(config_raw.get("metrics", {})),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    pd.Series(metadata, dtype=object).to_json(out_root / "unified_cv_run.json", indent=2)


def _write_optional_audit(
    *,
    out_root: Path,
    folds: Sequence[UnifiedCVFold],
    predictions: Sequence[UnifiedCVFoldPrediction],
) -> None:
    try:
        from .unified_audit import audit_unified_cv_predictions, write_unified_audit
    except Exception:
        return
    audit_df = audit_unified_cv_predictions(folds=folds, predictions=predictions)
    write_unified_audit(audit_df, out_root)


def _coerce_fold_prediction(
    fold: UnifiedCVFold,
    raw: UnifiedCVFoldPrediction | tuple[np.ndarray, np.ndarray] | Mapping[str, Any],
) -> UnifiedCVFoldPrediction:
    if isinstance(raw, UnifiedCVFoldPrediction):
        return raw
    if isinstance(raw, tuple):
        if len(raw) != 2:
            raise ValueError("fit_predict tuple output must be (val_pred_sec, test_pred_sec)")
        val_pred, test_pred = raw
        return UnifiedCVFoldPrediction(
            fold_id=fold.fold_id,
            val_pred_sec=np.asarray(val_pred, dtype=np.float32).reshape(-1),
            test_pred_sec=np.asarray(test_pred, dtype=np.float32).reshape(-1),
            val_meta=fold.val_meta,
            test_meta=fold.test_meta,
        )
    if isinstance(raw, Mapping):
        if "val_pred_sec" not in raw or "test_pred_sec" not in raw:
            raise ValueError("fit_predict mapping output must include val_pred_sec and test_pred_sec")
        return UnifiedCVFoldPrediction(
            fold_id=int(raw.get("fold_id", fold.fold_id)),
            val_pred_sec=np.asarray(raw["val_pred_sec"], dtype=np.float32).reshape(-1),
            test_pred_sec=np.asarray(raw["test_pred_sec"], dtype=np.float32).reshape(-1),
            val_meta=tuple(raw.get("val_meta", fold.val_meta)),
            test_meta=tuple(raw.get("test_meta", fold.test_meta)),
            model_info=dict(raw.get("model_info", {})),
        )
    raise TypeError(f"unsupported fit_predict output type: {type(raw)!r}")


def _validate_prediction_lengths(fold: UnifiedCVFold, pred: UnifiedCVFoldPrediction) -> None:
    if int(pred.fold_id) != int(fold.fold_id):
        raise ValueError(f"prediction fold_id mismatch: got={pred.fold_id}, expected={fold.fold_id}")
    if len(pred.val_pred_sec) != len(fold.y_val_sec):
        raise ValueError(f"validation prediction length mismatch: got={len(pred.val_pred_sec)}, expected={len(fold.y_val_sec)}")
    if len(pred.test_pred_sec) != len(fold.y_test_sec):
        raise ValueError(f"test prediction length mismatch: got={len(pred.test_pred_sec)}, expected={len(fold.y_test_sec)}")
    if len(pred.val_meta) != len(fold.val_meta):
        raise ValueError(f"validation metadata length mismatch: got={len(pred.val_meta)}, expected={len(fold.val_meta)}")
    if len(pred.test_meta) != len(fold.test_meta):
        raise ValueError(f"test metadata length mismatch: got={len(pred.test_meta)}, expected={len(fold.test_meta)}")


def _collect_split_parts(
    mats: Mapping[str, Any],
    dataset_ids: tuple[str, ...],
    dataset_indices: Mapping[str, DatasetFoldIndices],
    split_name: SplitName,
) -> dict[str, list[Any]]:
    parts: dict[str, list[Any]] = {"X": [], "X_mol": [], "X_cp": [], "y_sec": [], "meta": []}
    for dataset_id in dataset_ids:
        mat = mats[dataset_id]
        indices = dataset_indices[dataset_id]
        row_idx = getattr(indices, f"{split_name}_idx")
        ids = list(getattr(mat, "ids", [str(i) for i in range(len(mat.y_sec))]))
        for attr_name in ("X", "X_mol", "X_cp", "y_sec"):
            arr = np.asarray(getattr(mat, attr_name))
            parts[attr_name].append(arr[np.asarray(row_idx, dtype=int)])
        for local_idx in np.asarray(row_idx, dtype=int).tolist():
            parts["meta"].append(
                UnifiedCVRowMeta(
                    dataset_id=dataset_id,
                    original_row_id=str(ids[int(local_idx)]),
                    local_row_index=int(local_idx),
                    split_name=split_name,
                    fold_id=int(indices.fold_id),
                )
            )
    return parts


def _concat_arrays(parts: list[Any]) -> np.ndarray:
    if not parts:
        return np.asarray([], dtype=np.float32)
    arrays = [np.asarray(part) for part in parts]
    if len(arrays) == 1:
        return arrays[0].copy()
    return np.concatenate(arrays, axis=0)

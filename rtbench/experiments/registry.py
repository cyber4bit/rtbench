from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .archive import (
    ConfigCatalog,
    _canonical_config_sha1,
    _clean_json_value,
    _file_sha1,
    _normalize_relpath,
    _read_yaml,
    archive_effective_config,
    build_config_catalog,
    resolve_catalog_entry,
)


REGISTRY_COLUMNS = [
    "experiment_name",
    "run_dir",
    "output_root",
    "run_date",
    "status",
    "cleanable",
    "archived",
    "archived_note",
    "config_sha1",
    "config_hash_type",
    "config_path",
    "config_source",
    "effective_config_path",
    "effective_config_sha1",
    "summary_path",
    "avg_mae",
    "avg_r2",
    "win_both",
    "dataset_count",
    "seed_count",
    "key_hparams",
    "error",
]

KEY_HPARAM_PATHS = [
    "data.cpvec.enabled",
    "split.strategy",
    "models.ONLY_HYPER_TL",
    "models.ENABLE_HYPER_TL",
    "models.ENABLE_FAIL_TUNING",
    "models.FUSION_TOP_K",
    "models.CALIBRATE",
    "models.CLIP_MULT",
    "models.ENABLE_MDL_SUBSET_CANDIDATES",
    "models.ENABLE_LOCAL_TRANSFORM_CANDIDATES",
    "models.LOCAL_TARGET_TRANSFORMS",
    "models.HYPER_TL.balance_pretrain_by_dataset",
    "models.HYPER_TL.use_mdl_subset_mol",
    "models.HYPER_TL.n_models",
    "models.HYPER_TL.embed_dim",
    "models.HYPER_TL.mol_hidden",
    "models.HYPER_TL.cp_hidden",
    "models.HYPER_TL.dropout",
    "transfer_weights.source",
    "transfer_weights.target",
    "transfer_weights.adaptive_source",
    "transfer_weights.source_weight_mode",
    "transfer_weights.top_k_sources",
    "transfer_weights.target_transform",
    "transfer_weights.target_normalize",
    "transfer_weights.similarity_power",
    "transfer_weights.min_scale",
    "transfer_weights.max_scale",
    "seeds.default",
]

DEFAULT_REGISTRY = Path("experiments/registry.csv")
DEFAULT_CLEANUP_MANIFEST = Path("experiments/cleanup_candidates.txt")


@dataclass(frozen=True)
class MigrationSummary:
    record_count: int
    output_root_count: int
    cleanable_root_count: int
    registry_path: Path
    cleanup_manifest_path: Path


def _path_from_env(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


def _registry_cli_default() -> str:
    env_path = _path_from_env("RTBENCH_REGISTRY_PATH")
    return str(env_path) if env_path is not None else str(DEFAULT_REGISTRY)


def _resolve_project_path(project_root: Path, path: str | Path | None, default: Path) -> Path:
    selected = Path(path) if path is not None else default
    return selected if selected.is_absolute() else project_root / selected


def _resolve_registry_path(project_root: Path, registry_path: str | Path | None = None) -> Path:
    selected = Path(registry_path) if registry_path is not None else (_path_from_env("RTBENCH_REGISTRY_PATH") or DEFAULT_REGISTRY)
    return selected if selected.is_absolute() else project_root / selected


def _is_outputs_root_name(name: str) -> bool:
    return name == "outputs" or name.startswith("outputs_")


def _get_nested(raw: dict[str, Any], dotted: str) -> Any:
    cur: Any = raw
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _format_float(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.6f}"


def _format_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(int(value))


def _bool_to_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _serialize_hparams(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    return json.dumps(_clean_json_value(payload), ensure_ascii=True, sort_keys=True)


def _load_run_resolved_config(project_root: Path, run_dir_abs: Path) -> tuple[dict[str, Any] | None, str, str]:
    resolved_path = run_dir_abs / "config.resolved.yaml"
    if not resolved_path.exists():
        return None, "", ""
    raw = _read_yaml(resolved_path)
    if raw is None:
        return None, "", ""
    try:
        relpath = _normalize_relpath(resolved_path.relative_to(project_root))
    except ValueError:
        relpath = _normalize_relpath(resolved_path)
    return raw, relpath, _file_sha1(resolved_path)


def _extract_key_hparams_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for dotted in KEY_HPARAM_PATHS:
        value = _get_nested(raw, dotted)
        if value is None:
            continue
        payload[dotted] = value

    external = _get_nested(raw, "datasets.external")
    if isinstance(external, list):
        ds = [str(x).zfill(4) for x in external]
        payload["datasets.external_count"] = len(ds)
        if len(ds) <= 5:
            payload["datasets.external"] = ds

    pretrain = _get_nested(raw, "datasets.pretrain")
    if isinstance(pretrain, list):
        payload["datasets.pretrain_count"] = len(pretrain)

    return payload


def _infer_hparams_from_run_dir(run_dir: str) -> dict[str, Any]:
    text = run_dir.lower()
    parts = [p for p in run_dir.split("/") if p]
    payload: dict[str, Any] = {}

    if "cpvec" in text:
        payload["data.cpvec.enabled"] = True
    if "log1p" in text:
        payload["transfer_weights.target_transform"] = "log1p"
    elif "logk" in text:
        payload["transfer_weights.target_transform"] = "logk"
    elif "gradnorm" in text:
        payload["transfer_weights.target_transform"] = "gradient_norm"

    if "calibfalse" in text:
        payload["models.CALIBRATE"] = False
    elif "calibrated" in text or "calibtrue" in text:
        payload["models.CALIBRATE"] = True

    if "randomsplit" in text or "_random_" in text:
        payload["split.strategy"] = "random"

    if "hyper" in text:
        payload["models.ENABLE_HYPER_TL"] = True
    if "hybrid" in text:
        payload["models.ONLY_HYPER_TL"] = False
    if "trees" in text and "hyper" not in text:
        payload["models.ENABLE_HYPER_TL"] = False

    match = re.search(r"ens(\d+)", text)
    if match:
        payload["models.HYPER_TL.n_models"] = int(match.group(1))

    match = re.search(r"top(\d+)", text)
    if match:
        payload["transfer_weights.top_k_sources"] = int(match.group(1))

    match = re.search(r"seeds(\d+)_(\d+)", text)
    if match:
        payload["seeds.default"] = f"{match.group(1)}:{match.group(2)}"

    if parts and re.fullmatch(r"\d{4}", parts[-1]):
        payload["datasets.external"] = [parts[-1]]
        payload["datasets.external_count"] = 1
    if len(parts) >= 2 and re.fullmatch(r"S\d+", parts[1], flags=re.IGNORECASE):
        payload["supp.sheet"] = parts[1].upper()

    return payload


def _runtime_scope_hparams(run_dir: str) -> dict[str, Any]:
    parts = [p for p in run_dir.split("/") if p]
    payload: dict[str, Any] = {}
    if parts and re.fullmatch(r"\d{4}", parts[-1]):
        payload["run.target_dataset"] = parts[-1]
        payload["datasets.external"] = [parts[-1]]
        payload["datasets.external_count"] = 1
    if len(parts) >= 2 and re.fullmatch(r"S\d+", parts[1], flags=re.IGNORECASE):
        payload["run.sheet"] = parts[1].upper()
    return payload


def _summary_metrics_from_df(summary_df: pd.DataFrame, per_seed_path: Path | None) -> dict[str, str]:
    metrics = {
        "summary_path": "",
        "avg_mae": "",
        "avg_r2": "",
        "win_both": "",
        "dataset_count": "",
        "seed_count": "",
    }
    if summary_df.empty:
        return metrics

    if "our_mae_mean" in summary_df.columns:
        metrics["avg_mae"] = _format_float(float(summary_df["our_mae_mean"].mean()))
    if "our_r2_mean" in summary_df.columns:
        metrics["avg_r2"] = _format_float(float(summary_df["our_r2_mean"].mean()))
    if "dataset" in summary_df.columns:
        ds = summary_df["dataset"].astype(str).str.zfill(4).dropna().unique().tolist()
        metrics["dataset_count"] = _format_int(len(ds))
    else:
        metrics["dataset_count"] = _format_int(int(summary_df.shape[0]))

    if "win_both" in summary_df.columns:
        win_mask = summary_df["win_both"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
        metrics["win_both"] = _format_int(int(win_mask.sum()))

    if per_seed_path is not None and per_seed_path.exists():
        try:
            per_seed_df = pd.read_csv(per_seed_path, dtype={"dataset": str}, encoding="utf-8")
        except Exception:
            per_seed_df = pd.DataFrame()
        if not per_seed_df.empty and "seed" in per_seed_df.columns:
            seed_count = pd.Series(per_seed_df["seed"]).dropna().nunique()
            metrics["seed_count"] = _format_int(int(seed_count))

    return metrics


def _load_summary_metrics(run_dir_abs: Path, summary_df: pd.DataFrame | None = None) -> tuple[dict[str, str], str]:
    summary_path = run_dir_abs / "metrics" / "summary_vs_paper.csv"
    per_seed_path = run_dir_abs / "metrics" / "per_seed.csv"
    if summary_df is None:
        if not summary_path.exists():
            return _summary_metrics_from_df(pd.DataFrame(), per_seed_path), ""
        try:
            summary_df = pd.read_csv(summary_path, dtype={"dataset": str}, encoding="utf-8")
        except Exception:
            summary_df = pd.DataFrame()
    metrics = _summary_metrics_from_df(summary_df, per_seed_path)
    return metrics, (_normalize_relpath(summary_path) if summary_path.exists() else "")


def _default_run_date(run_dir_abs: Path) -> str:
    candidates = [
        run_dir_abs / "metrics" / "summary_vs_paper.csv",
        run_dir_abs / "config.sha1",
        run_dir_abs,
    ]
    for path in candidates:
        if path.exists():
            ts = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
            return ts.replace(microsecond=0).isoformat()
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def status_for_run_dir(run_dir: str | Path, has_summary: bool) -> str:
    rel = _normalize_relpath(run_dir)
    top = rel.split("/", 1)[0] if rel else ""
    if top.startswith("outputs_tmp"):
        return "tmp"
    return "success" if has_summary else "failed"


def _ensure_registry_frame(registry_path: Path) -> pd.DataFrame:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        try:
            df = pd.read_csv(registry_path, dtype=str, encoding="utf-8").fillna("")
        except Exception:
            df = pd.DataFrame(columns=REGISTRY_COLUMNS)
    else:
        df = pd.DataFrame(columns=REGISTRY_COLUMNS)
    for col in REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[REGISTRY_COLUMNS].copy()


def _write_registry(registry_path: Path, records: list[dict[str, str]], replace: bool) -> None:
    df = _ensure_registry_frame(registry_path)
    if replace:
        out_df = pd.DataFrame(records, columns=REGISTRY_COLUMNS)
    else:
        add_df = pd.DataFrame(records, columns=REGISTRY_COLUMNS)
        out_df = pd.concat([df, add_df], ignore_index=True)
        out_df = out_df.drop_duplicates(subset=["run_dir"], keep="last")
    if out_df.empty:
        out_df = pd.DataFrame(columns=REGISTRY_COLUMNS)
    else:
        out_df = out_df.fillna("").sort_values(["run_dir", "run_date"]).reset_index(drop=True)
    out_df.to_csv(registry_path, index=False, encoding="utf-8")


def _build_registry_record(
    project_root: Path,
    run_dir_abs: Path,
    *,
    catalog: ConfigCatalog,
    config_sha1: str | None = None,
    config_hash_type: str = "",
    config_path: str | Path | None = None,
    config_raw: dict[str, Any] | None = None,
    config_source: str = "",
    summary_df: pd.DataFrame | None = None,
    status: str | None = None,
    error: str = "",
    extra_hparams: dict[str, Any] | None = None,
    run_date: str | None = None,
) -> dict[str, str]:
    run_dir_rel = _normalize_relpath(run_dir_abs.relative_to(project_root))
    output_root = run_dir_rel.split("/", 1)[0]
    sha_path = run_dir_abs / "config.sha1"
    resolved_sha1 = str(config_sha1 or "").strip()
    if not resolved_sha1 and sha_path.exists():
        resolved_sha1 = sha_path.read_text(encoding="utf-8").strip()

    resolved_entry = None
    resolved_source = config_source
    resolved_hash_type = config_hash_type
    if config_raw is None and config_path is None:
        resolved_entry, resolved_source, guessed_hash_type = resolve_catalog_entry(run_dir_rel, resolved_sha1, catalog)
        if not resolved_hash_type:
            resolved_hash_type = guessed_hash_type
    if config_path is None and resolved_entry is not None:
        config_path = resolved_entry.config_path
    if config_raw is None and resolved_entry is not None:
        config_raw = resolved_entry.raw

    effective_config_raw, effective_config_path, effective_config_sha1 = _load_run_resolved_config(project_root, run_dir_abs)
    runtime_config_raw = effective_config_raw or config_raw
    if effective_config_raw is not None:
        if not resolved_source:
            resolved_source = "run_snapshot"
    elif config_raw is not None:
        effective_config_path, effective_config_sha1 = archive_effective_config(
            project_root,
            config_raw=config_raw,
            config_sha1=resolved_sha1,
        )
        if not resolved_sha1:
            resolved_sha1 = effective_config_sha1

    metrics, summary_relpath = _load_summary_metrics(run_dir_abs, summary_df=summary_df)
    if summary_relpath:
        try:
            summary_relpath = _normalize_relpath((run_dir_abs / "metrics" / "summary_vs_paper.csv").relative_to(project_root))
        except ValueError:
            summary_relpath = _normalize_relpath(run_dir_abs / "metrics" / "summary_vs_paper.csv")
    has_summary = bool(summary_relpath)
    if not resolved_source and resolved_entry is not None:
        resolved_source = "catalog"
    if not resolved_hash_type and resolved_sha1 and runtime_config_raw is not None:
        normalized_sha1 = _canonical_config_sha1(runtime_config_raw)
        if resolved_sha1 == normalized_sha1:
            resolved_hash_type = "normalized_sha1"

    hparams: dict[str, Any] = {}
    if runtime_config_raw is not None:
        hparams.update(_extract_key_hparams_from_raw(runtime_config_raw))
    hparams.update(_runtime_scope_hparams(run_dir_rel))
    if extra_hparams:
        hparams.update(extra_hparams)
    if not hparams:
        hparams.update(_infer_hparams_from_run_dir(run_dir_rel))

    config_path_text = _normalize_relpath(config_path) if config_path else ""
    run_date_text = run_date or _default_run_date(run_dir_abs)
    status_text = status or status_for_run_dir(run_dir_rel, has_summary=has_summary)
    archived_flag = bool(resolved_entry.archived) if resolved_entry is not None else False
    archived_note = str(resolved_entry.archived_note).strip() if resolved_entry is not None else ""

    return {
        "experiment_name": run_dir_rel,
        "run_dir": run_dir_rel,
        "output_root": output_root,
        "run_date": run_date_text,
        "status": status_text,
        "cleanable": _bool_to_text(output_root.startswith("outputs_tmp")),
        "archived": _bool_to_text(archived_flag),
        "archived_note": archived_note,
        "config_sha1": resolved_sha1,
        "config_hash_type": resolved_hash_type,
        "config_path": config_path_text,
        "config_source": resolved_source,
        "effective_config_path": effective_config_path,
        "effective_config_sha1": effective_config_sha1,
        "summary_path": summary_relpath,
        "avg_mae": metrics["avg_mae"],
        "avg_r2": metrics["avg_r2"],
        "win_both": metrics["win_both"],
        "dataset_count": metrics["dataset_count"],
        "seed_count": metrics["seed_count"],
        "key_hparams": _serialize_hparams(hparams),
        "error": str(error or "").strip(),
    }


def discover_run_dirs(project_root: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for sha_path in project_root.rglob("config.sha1"):
        rel = sha_path.relative_to(project_root)
        if rel.parts and _is_outputs_root_name(rel.parts[0]):
            run_dirs.add(sha_path.parent)

    for summary_path in project_root.rglob("summary_vs_paper.csv"):
        rel = summary_path.relative_to(project_root)
        if len(rel.parts) >= 3 and rel.parts[-2] == "metrics" and _is_outputs_root_name(rel.parts[0]):
            run_dirs.add(summary_path.parent.parent)

    return sorted(run_dirs, key=lambda p: _normalize_relpath(p.relative_to(project_root)))


def list_tmp_cleanup_candidates(project_root: Path) -> list[Path]:
    return sorted([p for p in project_root.iterdir() if p.is_dir() and p.name.startswith("outputs_tmp")], key=lambda p: p.name)


def write_cleanup_manifest(project_root: Path, manifest_path: Path) -> list[Path]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = list_tmp_cleanup_candidates(project_root)
    lines = [p.name for p in candidates]
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return candidates


def cleanup_tmp_outputs(project_root: Path, *, delete: bool = False) -> list[Path]:
    candidates = list_tmp_cleanup_candidates(project_root)
    if delete:
        root_resolved = project_root.resolve()
        for path in candidates:
            target = path.resolve()
            if target.parent != root_resolved or not path.name.startswith("outputs_tmp"):
                raise ValueError(f"Refusing to delete unexpected path: {path}")
            shutil.rmtree(path)
    return candidates


class ExperimentRegistry:
    def __init__(self, project_root: Path, *, registry_path: Path | None = None) -> None:
        project_root_path = Path(project_root)
        project_root_abs = project_root_path if project_root_path.is_absolute() else Path.cwd() / project_root_path
        self.project_root = project_root_abs.resolve()
        self.registry_path = _resolve_registry_path(project_root_abs, registry_path)

    @property
    def path(self) -> Path:
        return self.registry_path

    def load(self, *, refresh: bool = False) -> tuple[pd.DataFrame, Path]:
        if refresh or not self.registry_path.exists():
            self.migrate()
        return _ensure_registry_frame(self.registry_path), self.registry_path

    def replace(self, records: list[dict[str, str]]) -> None:
        _write_registry(self.registry_path, records, replace=True)

    def upsert(self, record: dict[str, str]) -> None:
        _write_registry(self.registry_path, [record], replace=False)

    def migrate(self, *, cleanup_manifest_path: Path | None = None) -> MigrationSummary:
        cleanup_manifest_path = _resolve_project_path(self.project_root, cleanup_manifest_path, DEFAULT_CLEANUP_MANIFEST)
        catalog = build_config_catalog(self.project_root)
        run_dirs = discover_run_dirs(self.project_root)
        records = [_build_registry_record(self.project_root, run_dir, catalog=catalog) for run_dir in run_dirs]
        self.replace(records)
        cleanable = write_cleanup_manifest(self.project_root, cleanup_manifest_path)
        output_roots = {record["output_root"] for record in records}
        return MigrationSummary(
            record_count=len(records),
            output_root_count=len(output_roots),
            cleanable_root_count=len(cleanable),
            registry_path=self.registry_path,
            cleanup_manifest_path=cleanup_manifest_path,
        )

    def record(
        self,
        *,
        run_dir: str | Path,
        status: str,
        config_sha1: str = "",
        config_hash_type: str = "",
        config_path: str | Path | None = None,
        config_raw: dict[str, Any] | None = None,
        config_source: str = "runtime",
        summary_df: pd.DataFrame | None = None,
        error: str = "",
        extra_hparams: dict[str, Any] | None = None,
        run_date: str | None = None,
    ) -> None:
        run_dir_abs = (self.project_root / Path(run_dir)).resolve() if not Path(run_dir).is_absolute() else Path(run_dir).resolve()
        catalog = build_config_catalog(self.project_root)
        record = _build_registry_record(
            self.project_root,
            run_dir_abs,
            catalog=catalog,
            config_sha1=config_sha1,
            config_hash_type=config_hash_type,
            config_path=config_path,
            config_raw=config_raw,
            config_source=config_source,
            summary_df=summary_df,
            status=status,
            error=error,
            extra_hparams=extra_hparams,
            run_date=run_date or datetime.now().astimezone().replace(microsecond=0).isoformat(),
        )
        self.upsert(record)


def migrate_registry(
    project_root: Path,
    *,
    registry_path: Path | None = None,
    cleanup_manifest_path: Path | None = None,
) -> MigrationSummary:
    registry = ExperimentRegistry(project_root, registry_path=registry_path)
    return registry.migrate(cleanup_manifest_path=cleanup_manifest_path)


def record_experiment(
    project_root: Path,
    *,
    run_dir: str | Path,
    status: str,
    config_sha1: str = "",
    config_hash_type: str = "",
    config_path: str | Path | None = None,
    config_raw: dict[str, Any] | None = None,
    config_source: str = "runtime",
    summary_df: pd.DataFrame | None = None,
    error: str = "",
    extra_hparams: dict[str, Any] | None = None,
    registry_path: Path | None = None,
    run_date: str | None = None,
) -> None:
    registry = ExperimentRegistry(project_root, registry_path=registry_path)
    registry.record(
        run_dir=run_dir,
        status=status,
        config_sha1=config_sha1,
        config_hash_type=config_hash_type,
        config_path=config_path,
        config_raw=config_raw,
        config_source=config_source,
        summary_df=summary_df,
        error=error,
        extra_hparams=extra_hparams,
        run_date=run_date,
    )


def load_registry(
    project_root: Path,
    *,
    registry_path: Path | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, Path]:
    registry = ExperimentRegistry(project_root, registry_path=registry_path)
    return registry.load(refresh=refresh)


__all__ = [
    "DEFAULT_CLEANUP_MANIFEST",
    "DEFAULT_REGISTRY",
    "ExperimentRegistry",
    "MigrationSummary",
    "REGISTRY_COLUMNS",
    "cleanup_tmp_outputs",
    "discover_run_dirs",
    "list_tmp_cleanup_candidates",
    "load_registry",
    "migrate_registry",
    "record_experiment",
    "status_for_run_dir",
    "write_cleanup_manifest",
]

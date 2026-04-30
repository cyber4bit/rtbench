from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config import resolve_config


DEFAULT_CONFIG_ARCHIVE_DIR = Path("experiments/configs")
DEFAULT_ARCHIVED_CONFIG_MANIFEST = Path("experiments/archived_configs.csv")


@dataclass(frozen=True)
class ConfigCatalogEntry:
    config_path: str
    output_root: str
    file_sha1: str
    normalized_sha1: str
    raw: dict[str, Any]
    archived: bool = False
    archived_note: str = ""


@dataclass(frozen=True)
class ArchivedConfigAlias:
    archived_config_path: str
    archived_file_sha1: str
    canonical_config_path: str
    canonical_normalized_sha1: str
    note: str


@dataclass(frozen=True)
class ConfigCatalog:
    by_output_root: dict[str, ConfigCatalogEntry]
    by_file_sha1: dict[str, ConfigCatalogEntry]
    by_normalized_sha1: dict[str, ConfigCatalogEntry]
    by_config_path: dict[str, ConfigCatalogEntry]


def _normalize_relpath(path: str | Path) -> str:
    text = str(path).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def _canonical_config_sha1(raw: dict[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _file_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _get_nested(raw: dict[str, Any], dotted: str) -> Any:
    cur: Any = raw
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json_value(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_clean_json_value(v) for v in value]
    return value


def _yaml_text(raw: dict[str, Any]) -> str:
    return yaml.safe_dump(_clean_json_value(raw), sort_keys=False, allow_unicode=False)


def _write_text_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == text:
            return
    path.write_text(text, encoding="utf-8", newline="\n")


def _read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _load_archived_config_aliases(project_root: Path) -> list[ArchivedConfigAlias]:
    manifest_path = project_root / DEFAULT_ARCHIVED_CONFIG_MANIFEST
    if not manifest_path.exists():
        return []
    try:
        df = pd.read_csv(manifest_path, dtype=str, encoding="utf-8").fillna("")
    except Exception:
        return []
    required = {
        "archived_config_path",
        "archived_file_sha1",
        "canonical_config_path",
        "canonical_normalized_sha1",
        "note",
    }
    if not required.issubset(set(df.columns)):
        return []
    aliases: list[ArchivedConfigAlias] = []
    for _, row in df.iterrows():
        aliases.append(
            ArchivedConfigAlias(
                archived_config_path=_normalize_relpath(str(row["archived_config_path"])),
                archived_file_sha1=str(row["archived_file_sha1"]).strip(),
                canonical_config_path=_normalize_relpath(str(row["canonical_config_path"])),
                canonical_normalized_sha1=str(row["canonical_normalized_sha1"]).strip(),
                note=str(row["note"]).strip(),
            )
        )
    return aliases


def build_config_catalog(project_root: Path) -> ConfigCatalog:
    by_output_root: dict[str, ConfigCatalogEntry] = {}
    by_file_sha1: dict[str, ConfigCatalogEntry] = {}
    by_normalized_sha1: dict[str, ConfigCatalogEntry] = {}
    by_config_path: dict[str, ConfigCatalogEntry] = {}

    cfg_root = project_root / "configs"
    if not cfg_root.exists():
        return ConfigCatalog(by_output_root, by_file_sha1, by_normalized_sha1, by_config_path)

    for path in sorted(cfg_root.rglob("*.yaml")):
        try:
            resolved = resolve_config(path)
        except Exception:
            continue
        raw = resolved.raw
        output_root = _normalize_relpath(_get_nested(raw, "outputs.root") or "")
        rel_config = _normalize_relpath(path.relative_to(project_root))
        entry = ConfigCatalogEntry(
            config_path=rel_config,
            output_root=output_root,
            file_sha1=_file_sha1(path),
            normalized_sha1=_canonical_config_sha1(raw),
            raw=raw,
        )
        if output_root and output_root not in by_output_root:
            by_output_root[output_root] = entry
        by_file_sha1.setdefault(entry.file_sha1, entry)
        by_normalized_sha1.setdefault(entry.normalized_sha1, entry)
        by_config_path[entry.config_path] = entry

    snapshot_root = project_root / DEFAULT_CONFIG_ARCHIVE_DIR
    if snapshot_root.exists():
        for path in sorted(snapshot_root.glob("*.yaml")):
            raw = _read_yaml(path)
            if raw is None:
                continue
            entry = ConfigCatalogEntry(
                config_path=_normalize_relpath(path.relative_to(project_root)),
                output_root=_normalize_relpath(_get_nested(raw, "outputs.root") or ""),
                file_sha1=_file_sha1(path),
                normalized_sha1=_canonical_config_sha1(raw),
                raw=raw,
            )
            by_normalized_sha1.setdefault(entry.normalized_sha1, entry)

    for alias in _load_archived_config_aliases(project_root):
        entry = by_config_path.get(alias.canonical_config_path)
        if entry is None and alias.canonical_normalized_sha1:
            entry = by_normalized_sha1.get(alias.canonical_normalized_sha1)
        if entry is None:
            continue
        archived_entry = ConfigCatalogEntry(
            config_path=alias.archived_config_path,
            output_root=entry.output_root,
            file_sha1=alias.archived_file_sha1 or entry.file_sha1,
            normalized_sha1=entry.normalized_sha1,
            raw=entry.raw,
            archived=True,
            archived_note=alias.note or f"Archived duplicate of {entry.config_path}",
        )
        if archived_entry.file_sha1:
            by_file_sha1[archived_entry.file_sha1] = archived_entry
        if archived_entry.config_path:
            by_config_path[archived_entry.config_path] = archived_entry
    return ConfigCatalog(by_output_root, by_file_sha1, by_normalized_sha1, by_config_path)


def _guess_catalog_entry(run_dir: str, catalog: ConfigCatalog) -> tuple[ConfigCatalogEntry | None, str, str]:
    parts = [p for p in run_dir.split("/") if p]
    if not parts:
        return None, "", ""
    top = parts[0]
    versions = [token[1:] for token in top.split("_") if token.startswith("v") and token[1:].isdigit()]
    if top.startswith("outputs_supp_eval_") and versions:
        guess_path = f"configs/supp_eval_single_task_v{versions[-1]}.yaml"
        entry = catalog.by_config_path.get(guess_path)
        if entry is not None:
            return entry, "heuristic_version_match", ""
    return None, "", ""


def resolve_catalog_entry(
    run_dir: str,
    config_sha1: str,
    catalog: ConfigCatalog,
) -> tuple[ConfigCatalogEntry | None, str, str]:
    if config_sha1:
        entry = catalog.by_file_sha1.get(config_sha1)
        if entry is not None:
            return entry, "config_hash_match", "file_sha1"

    entry = catalog.by_output_root.get(run_dir)
    if entry is not None:
        hash_type = ""
        if config_sha1 and config_sha1 == entry.file_sha1:
            hash_type = "file_sha1"
        elif config_sha1 and config_sha1 == entry.normalized_sha1:
            hash_type = "normalized_sha1"
        return entry, "output_root_match", hash_type

    if config_sha1:
        entry = catalog.by_normalized_sha1.get(config_sha1)
        if entry is not None:
            return entry, "config_hash_match", "normalized_sha1"

    return _guess_catalog_entry(run_dir, catalog)


def archive_effective_config(
    project_root: Path,
    *,
    config_raw: dict[str, Any],
    config_sha1: str = "",
) -> tuple[str, str]:
    del config_sha1
    resolved_sha1 = _canonical_config_sha1(config_raw)
    archive_rel = DEFAULT_CONFIG_ARCHIVE_DIR / f"{resolved_sha1}.yaml"
    _write_text_if_changed(project_root / archive_rel, _yaml_text(config_raw))
    return _normalize_relpath(archive_rel), resolved_sha1


def effective_config_file_sha1(config_raw: dict[str, Any]) -> str:
    return _text_sha1(_yaml_text(config_raw))


def write_effective_config_snapshot(
    project_root: Path,
    *,
    run_dir: str | Path,
    config_raw: dict[str, Any],
    config_sha1: str = "",
    write_run_sha1: bool = False,
) -> tuple[str, str]:
    archive_path, resolved_sha1 = archive_effective_config(
        project_root,
        config_raw=config_raw,
        config_sha1=config_sha1,
    )
    run_dir_abs = (project_root / Path(run_dir)).resolve() if not Path(run_dir).is_absolute() else Path(run_dir).resolve()
    resolved_text = _yaml_text(config_raw)
    _write_text_if_changed(run_dir_abs / "config.resolved.yaml", resolved_text)
    if write_run_sha1:
        _write_text_if_changed(run_dir_abs / "config.sha1", _text_sha1(resolved_text))
    return archive_path, resolved_sha1


__all__ = [
    "ArchivedConfigAlias",
    "ConfigCatalog",
    "ConfigCatalogEntry",
    "DEFAULT_ARCHIVED_CONFIG_MANIFEST",
    "DEFAULT_CONFIG_ARCHIVE_DIR",
    "archive_effective_config",
    "build_config_catalog",
    "effective_config_file_sha1",
    "resolve_catalog_entry",
    "write_effective_config_snapshot",
]

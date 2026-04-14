from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rtbench.config import REQUIRED_CONFIG_SECTIONS, resolve_config


TOP_LEVEL_ORDER = [
    "data",
    "datasets",
    "split",
    "models",
    "transfer_weights",
    "seeds",
    "metrics",
    "stats",
    "outputs",
]
BASE_DIR = Path("configs/_bases")
ARCHIVED_CONFIG_MANIFEST = Path("experiments/archived_configs.csv")
REPORT_PATH = Path("experiments/config_refactor_phase6_report.json")


@dataclass(frozen=True)
class ConfigSnapshot:
    path: Path
    raw: dict[str, Any]
    file_sha1: str
    normalized_sha1: str


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config '{path}' must contain a YAML mapping.")
    return raw


def _is_policy_config(raw: dict[str, Any]) -> bool:
    return bool({"runs", "policy"}.intersection(set(raw.keys()))) and not bool({"data", "datasets"}.intersection(set(raw.keys())))


def _normalize_relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _file_sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _normalized_sha1(raw: dict[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _set_nested(out: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = out
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = deepcopy(value)


def _extract_subset(raw: dict[str, Any], dotted_paths: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dotted in dotted_paths:
        cur: Any = raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"Missing expected path '{dotted}'")
            cur = cur[part]
        _set_nested(out, dotted, cur)
    return out


def _has_structure(raw: Any, template: Any) -> bool:
    if isinstance(template, dict):
        if not isinstance(raw, dict):
            return False
        return all(key in raw and _has_structure(raw[key], value) for key, value in template.items())
    return True


def _recursive_diff(base: Any, target: Any) -> Any:
    if isinstance(base, dict) and isinstance(target, dict):
        out: dict[str, Any] = {}
        for key, value in target.items():
            if key not in base:
                out[key] = deepcopy(value)
                continue
            nested = _recursive_diff(base[key], value)
            if nested not in ({}, []):
                out[key] = nested
        return out
    if target == base:
        return {}
    return deepcopy(target)


def _ordered_doc(base_refs: list[str], diff: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"_base": base_refs[0] if len(base_refs) == 1 else base_refs}
    for key in TOP_LEVEL_ORDER:
        if key in diff:
            out[key] = diff[key]
    for key, value in diff.items():
        if key not in out:
            out[key] = value
    return out


def _discover_benchmark_configs(project_root: Path) -> tuple[list[ConfigSnapshot], list[str]]:
    configs: list[ConfigSnapshot] = []
    skipped: list[str] = []
    for path in sorted((project_root / "configs").rglob("*.yaml")):
        if path.parent.name == "_bases":
            continue
        raw = _read_yaml(path)
        if _is_policy_config(raw):
            skipped.append(_normalize_relpath(path, project_root))
            continue
        try:
            resolved = resolve_config(path)
        except Exception as exc:
            skipped.append(f"{_normalize_relpath(path, project_root)} ({exc})")
            continue
        configs.append(
            ConfigSnapshot(
                path=path,
                raw=resolved.raw,
                file_sha1=_file_sha1(path),
                normalized_sha1=_normalized_sha1(resolved.raw),
            )
        )
    return configs, skipped


def _rank_duplicate_candidate(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    score = 0
    if name.startswith("tmp_"):
        score += 100
    if "smoke_tmp" in name:
        score += 50
    if "tmp_" in path.stem.lower():
        score += 10
    return score, len(path.name), path.as_posix()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    path.write_text(text, encoding="utf-8")


def _build_base_payloads(project_root: Path, snapshots: dict[str, ConfigSnapshot]) -> dict[str, dict[str, Any]]:
    rplc = snapshots["configs/rplc_14x14.yaml"].raw
    cpvec = snapshots["configs/rplc_14x14_cpvec_hyper_best_v2.yaml"].raw
    return {
        "_base_rplc_14x14.yaml": _extract_subset(
            rplc,
            [
                "data.repo_url",
                "data.commit",
                "data.local_root",
                "data.baseline_csv",
                "data.gradient_points",
                "datasets.pretrain",
                "datasets.expected_pretrain_count",
                "split.train",
                "split.val",
                "split.test",
                "metrics.paper_avg_mae",
                "metrics.paper_avg_r2",
                "metrics.required_win_both",
                "stats.test",
                "stats.correction",
                "stats.fdr_q",
            ],
        ),
        "_base_models_tree.yaml": _extract_subset(
            rplc,
            [
                "models.XGB_A",
                "models.XGB_B",
                "models.LGBM_A",
                "models.LGBM_B",
            ],
        ),
        "_base_transfer.yaml": _extract_subset(
            rplc,
            [
                "transfer_weights.source",
                "transfer_weights.target",
                "transfer_weights.adaptive_source",
            ],
        ),
        "_base_cpvec.yaml": _extract_subset(cpvec, ["data.cpvec"]),
    }


def _archive_duplicates(
    project_root: Path,
    duplicates: list[tuple[ConfigSnapshot, ConfigSnapshot]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for archived, canonical in duplicates:
        rows.append(
            {
                "archived_config_path": _normalize_relpath(archived.path, project_root),
                "archived_file_sha1": archived.file_sha1,
                "canonical_config_path": _normalize_relpath(canonical.path, project_root),
                "canonical_normalized_sha1": canonical.normalized_sha1,
                "note": f"Archived exact duplicate of {_normalize_relpath(canonical.path, project_root)}",
            }
        )
        if archived.path.exists():
            archived.path.unlink()
    manifest_path = project_root / ARCHIVED_CONFIG_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8")
    return rows


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    benchmark_configs, skipped_configs = _discover_benchmark_configs(project_root)
    snapshots = {_normalize_relpath(snapshot.path, project_root): snapshot for snapshot in benchmark_configs}

    duplicate_groups: dict[str, list[ConfigSnapshot]] = defaultdict(list)
    for snapshot in benchmark_configs:
        duplicate_groups[snapshot.normalized_sha1].append(snapshot)

    duplicates_to_archive: list[tuple[ConfigSnapshot, ConfigSnapshot]] = []
    archived_relpaths: set[str] = set()
    for group in duplicate_groups.values():
        if len(group) <= 1:
            continue
        ordered = sorted(group, key=lambda item: _rank_duplicate_candidate(item.path))
        canonical = ordered[0]
        for archived in ordered[1:]:
            duplicates_to_archive.append((archived, canonical))
            archived_relpaths.add(_normalize_relpath(archived.path, project_root))

    base_payloads = _build_base_payloads(project_root, snapshots)
    base_refs = {
        name: Path("_bases") / name for name in base_payloads
    }
    base_rplc = base_payloads["_base_rplc_14x14.yaml"]
    base_models_tree = base_payloads["_base_models_tree.yaml"]
    base_transfer = base_payloads["_base_transfer.yaml"]
    base_cpvec = base_payloads["_base_cpvec.yaml"]

    for name, payload in base_payloads.items():
        _write_yaml(project_root / BASE_DIR / name, payload)

    base_usage = Counter()
    rewritten = 0
    report_rows: list[dict[str, Any]] = []
    for snapshot in benchmark_configs:
        relpath = _normalize_relpath(snapshot.path, project_root)
        if relpath in archived_relpaths:
            continue

        selected_bases: list[str] = []
        if _has_structure(snapshot.raw, base_rplc):
            selected_bases.append(base_refs["_base_rplc_14x14.yaml"].as_posix())
        if _has_structure(snapshot.raw, base_transfer):
            selected_bases.append(base_refs["_base_transfer.yaml"].as_posix())
        if _has_structure(snapshot.raw, base_models_tree):
            selected_bases.append(base_refs["_base_models_tree.yaml"].as_posix())
        if _has_structure(snapshot.raw, base_cpvec):
            selected_bases.append(base_refs["_base_cpvec.yaml"].as_posix())
        if not selected_bases:
            raise RuntimeError(f"No compatible base found for {relpath}")

        merged_base: dict[str, Any] = {}
        for base_ref in selected_bases:
            merged_base = _merge_dicts(merged_base, base_payloads[Path(base_ref).name])

        diff = _recursive_diff(merged_base, snapshot.raw)
        if not isinstance(diff, dict):
            raise RuntimeError(f"Unexpected diff payload for {relpath}")
        doc = _ordered_doc(selected_bases, diff)
        before_size = snapshot.path.stat().st_size
        _write_yaml(snapshot.path, doc)
        after_size = snapshot.path.stat().st_size

        resolved_after = resolve_config(snapshot.path).raw
        if resolved_after != snapshot.raw:
            raise RuntimeError(f"Resolved config changed for {relpath}")

        rewritten += 1
        for base_ref in selected_bases:
            base_usage[Path(base_ref).name] += 1
        report_rows.append(
            {
                "config_path": relpath,
                "base_count": len(selected_bases),
                "bases": selected_bases,
                "before_size": before_size,
                "after_size": after_size,
                "normalized_sha1": snapshot.normalized_sha1,
            }
        )

    archived_rows = _archive_duplicates(project_root, duplicates_to_archive)

    report = {
        "benchmark_config_count": len(benchmark_configs),
        "policy_config_count": len(skipped_configs),
        "rewritten_config_count": rewritten,
        "archived_config_count": len(archived_rows),
        "base_usage": dict(base_usage),
        "archived_configs": archived_rows,
        "skipped_configs": skipped_configs,
        "configs": report_rows,
    }
    report_path = project_root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("benchmark_config_count", "rewritten_config_count", "archived_config_count", "base_usage")}, ensure_ascii=True))


if __name__ == "__main__":
    main()

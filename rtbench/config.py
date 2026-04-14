from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_CONFIG_SECTIONS = [
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


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]
    datasets: dict[str, Any]
    split: dict[str, Any]
    models: dict[str, Any]
    transfer_weights: dict[str, Any]
    seeds: dict[str, Any]
    metrics: dict[str, Any]
    stats: dict[str, Any]
    outputs: dict[str, Any]


@dataclass(frozen=True)
class ResolvedConfig:
    path: Path
    raw: dict[str, Any]
    config: Config
    base_chain: tuple[str, ...]
    overrides: tuple[str, ...]


def validate_config_dict(raw: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED_CONFIG_SECTIONS if k not in raw]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    return raw


def _config_from_raw(raw: dict[str, Any]) -> Config:
    validate_config_dict(raw)
    return Config(**{k: copy.deepcopy(raw[k]) for k in REQUIRED_CONFIG_SECTIONS})


def _read_yaml_dict(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config '{path}' must contain a YAML mapping at the top level")
    return raw


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _append_unique_paths(items: list[Path], paths: list[Path]) -> None:
    seen = {p.resolve() for p in items}
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            items.append(resolved)
            seen.add(resolved)


def _normalize_base_entries(base_value: Any, path: Path) -> list[Path]:
    if isinstance(base_value, (str, Path)):
        base_items = [base_value]
    elif isinstance(base_value, list):
        base_items = base_value
    else:
        raise ValueError(f"Config '{path}' has invalid _base value: {base_value!r}")

    resolved_paths = []
    for entry in base_items:
        base_path = Path(str(entry))
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        else:
            base_path = base_path.resolve()
        resolved_paths.append(base_path)
    return resolved_paths


def _resolve_config_recursive(path: Path, stack: tuple[Path, ...]) -> tuple[dict[str, Any], list[Path]]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(p) for p in stack + (path,))
        raise ValueError(f"Config inheritance cycle detected: {cycle}")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _read_yaml_dict(path)
    base_value = raw.pop("_base", None)
    merged: dict[str, Any] = {}
    chain: list[Path] = []
    if base_value is not None:
        for base_path in _normalize_base_entries(base_value, path):
            base_raw, base_chain = _resolve_config_recursive(base_path, stack + (path,))
            merged = _merge_dicts(merged, base_raw)
            _append_unique_paths(chain, base_chain)
    merged = _merge_dicts(merged, raw)
    _append_unique_paths(chain, [path])
    return merged, chain


def parse_override_value(text: str) -> Any:
    value_text = str(text).strip()
    lower = value_text.lower()
    if lower in {"true", "false", "null", "none"}:
        return yaml.safe_load(lower if lower != "none" else "null")
    if value_text.startswith(("[", "{", "\"", "'")):
        return yaml.safe_load(value_text)
    try:
        if any(ch in value_text for ch in (".", "e", "E")):
            return float(value_text)
        return int(value_text)
    except ValueError:
        return value_text


def parse_override_expr(expr: str) -> tuple[str, Any]:
    text = str(expr).strip()
    if "=" not in text:
        raise ValueError(f"Invalid override '{expr}'. Expected dotted.path=value")
    dotted_path, value_text = text.split("=", 1)
    dotted_path = dotted_path.strip()
    if not dotted_path or any(not part.strip() for part in dotted_path.split(".")):
        raise ValueError(f"Invalid override path '{dotted_path}'")
    return dotted_path, parse_override_value(value_text)


def apply_override(raw: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [part.strip() for part in dotted_path.split(".")]
    cur: dict[str, Any] = raw
    for part in parts[:-1]:
        next_value = cur.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cur[part] = next_value
        cur = next_value
    cur[parts[-1]] = value


def apply_overrides(raw: dict[str, Any], overrides: list[str] | tuple[str, ...]) -> dict[str, Any]:
    out = copy.deepcopy(raw)
    for expr in overrides:
        dotted_path, value = parse_override_expr(expr)
        apply_override(out, dotted_path, value)
    return out


def resolve_config(path: str | Path, overrides: list[str] | tuple[str, ...] | None = None) -> ResolvedConfig:
    config_path = Path(path).resolve()
    raw, chain = _resolve_config_recursive(config_path, stack=())
    override_items = tuple(str(x) for x in (overrides or ()))
    if override_items:
        raw = apply_overrides(raw, override_items)
    validate_config_dict(raw)
    return ResolvedConfig(
        path=config_path,
        raw=raw,
        config=_config_from_raw(raw),
        base_chain=tuple(str(p) for p in chain),
        overrides=override_items,
    )


def load_raw_config(path: str | Path, overrides: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    return resolve_config(path, overrides=overrides).raw


def load_config(path: str | Path, overrides: list[str] | tuple[str, ...] | None = None) -> Config:
    return resolve_config(path, overrides=overrides).config


def parse_seed_range(seed_expr: str) -> list[int]:
    text = seed_expr.strip()
    if ":" in text:
        left, right = text.split(":", 1)
        start = int(left)
        end = int(right)
        if end < start:
            raise ValueError(f"Invalid seed range: {text}")
        return list(range(start, end + 1))
    if "," in text:
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    return [int(text)]

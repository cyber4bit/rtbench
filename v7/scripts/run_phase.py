from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PHASE_MANIFESTS = {
    "V7_probe": Path("v7/configs/v7_unified.yaml"),
    "V7_validate": Path("v7/configs/v7_unified.yaml"),
    "V7_final": Path("v7/configs/v7_unified.yaml"),
}

PHASE_ALIASES: dict[str, dict[str, Any]] = {
    "V7_probe": {
        "description": "Single-seed v7 diagnostic probe.",
        "overrides": {
            "seeds.default": "70",
            "outputs.root_template": "outputs_v7_probe_{sheet}",
            "models.SHEET_UNIFIED_HYPER_TL.epochs": 30,
            "models.HYPER_TL.epochs": 30,
        },
        "guardrails": {"forbidden_seed_ranges": [[0, 69], [71, 90]]},
        "outputs": {"stamp": "v7/reports/_stamps/V7_probe.ok"},
    },
    "V7_final": {
        "description": "Locked v7 final evaluation.",
        "requires": ["v7/reports/_stamps/V7_validate.ok"],
        "overrides": {
            "seeds.default": "81:90",
            "outputs.root_template": "outputs_v7_final_{sheet}",
        },
        "guardrails": {"forbidden_seed_ranges": [[0, 80]]},
        "outputs": {"stamp": "v7/reports/_stamps/V7_final.ok"},
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def _merge_manifest(base: dict[str, Any], alias: dict[str, Any] | None, phase: str) -> dict[str, Any]:
    manifest = dict(base)
    manifest["phase"] = phase
    if not alias:
        return manifest
    if "description" in alias:
        manifest["description"] = alias["description"]
    if "requires" in alias:
        manifest["requires"] = list(alias["requires"])
    for section in ("overrides", "guardrails", "outputs"):
        if section not in alias:
            continue
        current = dict(manifest.get(section, {}) or {})
        current.update(dict(alias[section] or {}))
        manifest[section] = current
    return manifest


def _apply_dotted_override(raw: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [part.strip() for part in str(dotted_path).split(".")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid override path: {dotted_path!r}")
    cur = raw
    for part in parts[:-1]:
        child = cur.get(part)
        if not isinstance(child, dict):
            child = {}
            cur[part] = child
        cur = child
    cur[parts[-1]] = value


def _parse_seeds(expr: str) -> list[int]:
    text = str(expr).strip()
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start), int(end) + 1))
    if "," in text:
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    return [int(text)]


def _load_seed_band(repo_root: Path, phase: str) -> list[int]:
    seed_cfg = _load_yaml(repo_root / "v7/configs/seed_allocation.yaml")
    bands = seed_cfg.get("seed_bands", {})
    if phase not in bands:
        raise ValueError(f"Unknown v7 seed band: {phase}")
    return [int(x) for x in bands[phase].get("seeds", [])]


def _require_paths(repo_root: Path, paths: list[str]) -> None:
    missing = [p for p in paths if not (repo_root / p).exists()]
    if missing:
        raise FileNotFoundError(f"Required v7 input(s) missing: {missing}")


def _phase_config_overrides(manifest: dict[str, Any], sheet: str) -> dict[str, Any]:
    overrides = dict(manifest.get("overrides", {}) or {})
    reserved = {"seeds.default", "outputs.root_template", "sheet_overrides"}
    out = {str(k): v for k, v in overrides.items() if str(k) not in reserved}
    sheet_overrides = overrides.get("sheet_overrides", {})
    if isinstance(sheet_overrides, dict):
        cur = sheet_overrides.get(sheet, {})
        if isinstance(cur, dict):
            out.update({str(k): v for k, v in cur.items()})
    return out


def _generated_config(repo_root: Path, manifest: dict[str, Any], sheet: str) -> tuple[Path, dict[str, Any]]:
    phase = str(manifest["phase"])
    base_configs = dict(manifest.get("base_configs", {}) or {})
    if sheet not in base_configs:
        raise ValueError(f"Manifest {phase} is missing base config for {sheet}")
    raw = _load_yaml(repo_root / str(base_configs[sheet]))
    overrides = dict(manifest.get("overrides", {}) or {})
    seed_expr = str(overrides.get("seeds.default", ""))
    if not seed_expr:
        raise ValueError(f"Manifest {phase} is missing overrides.seeds.default")
    actual_seeds = _parse_seeds(seed_expr)
    expected_seeds = _load_seed_band(repo_root, phase)
    if actual_seeds != expected_seeds:
        raise ValueError(f"{phase} seed band mismatch: manifest={actual_seeds}, expected={expected_seeds}")
    for item in manifest.get("guardrails", {}).get("forbidden_seed_ranges", []) or []:
        start, end = int(item[0]), int(item[1])
        if any(start <= seed <= end for seed in actual_seeds):
            raise ValueError(f"{phase} is not allowed to use seeds in forbidden range {start}..{end}")

    root_template = str(overrides.get("outputs.root_template", ""))
    raw.setdefault("seeds", {})["default"] = seed_expr
    raw.setdefault("outputs", {})["root"] = root_template.format(sheet=sheet)
    for dotted_path, value in _phase_config_overrides(manifest, sheet).items():
        _apply_dotted_override(raw, dotted_path, value)
    out_dir = repo_root / str(manifest.get("outputs", {}).get("generated_config_dir", "v7/reports/_generated_configs"))
    return out_dir / f"{phase}_{sheet}.yaml", raw


def run_phase(
    repo_root: Path,
    phase: str,
    generate_config_only: bool,
    only_sheet: str | None,
    eval_datasets: str,
) -> list[Path]:
    if phase not in PHASE_MANIFESTS:
        raise ValueError(f"Unknown phase {phase}; expected one of {sorted(PHASE_MANIFESTS)}")
    base = _load_yaml(repo_root / PHASE_MANIFESTS[phase])
    manifest = _merge_manifest(base, PHASE_ALIASES.get(phase), phase)
    _require_paths(repo_root, [str(p) for p in manifest.get("requires", []) or []])

    sheets = [only_sheet.upper()] if only_sheet else ["S4", "S5"]
    generated_paths: list[Path] = []
    for sheet in sheets:
        if sheet not in ("S4", "S5"):
            raise ValueError(f"Invalid sheet: {sheet}")
        out_path, raw = _generated_config(repo_root, manifest, sheet)
        _write_yaml(out_path, raw)
        generated_paths.append(out_path)
        print(f"wrote {out_path}")

    if generate_config_only:
        return generated_paths

    for path in generated_paths:
        cmd = [sys.executable, "-m", "rtbench.run", "--config", str(path), "--no-download"]
        if eval_datasets:
            cmd.extend(["--eval-datasets", eval_datasets])
        print("running " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=repo_root, check=True)

    stamp = repo_root / str(manifest.get("outputs", {}).get("stamp", f"v7/reports/_stamps/{phase}.ok"))
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text("ok\n", encoding="utf-8")
    print(f"wrote {stamp}")
    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an RTBench v7 phase")
    parser.add_argument("--phase", required=True, choices=sorted(PHASE_MANIFESTS))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--generate-config-only", action="store_true")
    parser.add_argument("--only-sheet", choices=["S4", "S5", "s4", "s5"], default=None)
    parser.add_argument("--eval-datasets", default="")
    args = parser.parse_args()
    run_phase(
        Path(args.repo_root).resolve(),
        args.phase,
        bool(args.generate_config_only),
        args.only_sheet,
        str(args.eval_datasets).strip(),
    )


if __name__ == "__main__":
    main()

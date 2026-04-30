from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PHASE_MANIFESTS = {
    "L1_focal_3x": Path("v4/configs/v4_focal_3x.yaml"),
    "L5_validate": Path("v4/configs/v4_validate.yaml"),
    "L6_final": Path("v4/configs/v4_final_eval.yaml"),
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


def _apply_dotted_override(raw: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = [part.strip() for part in str(dotted_path).split(".")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid override path: {dotted_path!r}")
    cur = raw
    for part in parts[:-1]:
        next_value = cur.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cur[part] = next_value
        cur = next_value
    cur[parts[-1]] = value


def _parse_seeds(expr: str) -> list[int]:
    text = str(expr).strip()
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start), int(end) + 1))
    if "," in text:
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    return [int(text)]


def _load_seed_band(phase: str, repo_root: Path) -> list[int]:
    seed_cfg = _load_yaml(repo_root / "v4/configs/seed_allocation.yaml")
    bands = seed_cfg.get("seed_bands", {})
    if phase not in bands:
        raise ValueError(f"Unknown seed band for phase {phase}")
    return [int(x) for x in bands[phase].get("seeds", [])]


def _require_paths(repo_root: Path, paths: list[str]) -> None:
    missing = [p for p in paths if not (repo_root / p).exists()]
    if missing:
        raise FileNotFoundError(f"Required v4 input(s) missing: {missing}")


def _rules_for_sheet(repo_root: Path, rules_path: str, sheet: str) -> dict[str, float]:
    data = _load_yaml(repo_root / rules_path)
    rules = data.get("rules", {})
    sheet_rules = rules.get(sheet, {})
    if not isinstance(sheet_rules, dict):
        raise ValueError(f"rules.{sheet} must be a mapping")
    return {str(k).zfill(4): float(v) for k, v in sheet_rules.items()}


def _phase_weights(repo_root: Path, manifest: dict[str, Any], sheet: str) -> dict[str, float]:
    overrides = dict(manifest.get("overrides", {}) or {})
    fixed = overrides.get("transfer_weights.per_dataset_multiplier")
    if isinstance(fixed, dict):
        sheet_fixed = fixed.get(sheet, {})
        if not isinstance(sheet_fixed, dict):
            raise ValueError(f"Fixed multiplier for {sheet} must be a mapping")
        return {str(k).zfill(4): float(v) for k, v in sheet_fixed.items()}

    rules_from = overrides.get("transfer_weights.per_dataset_multiplier_from")
    if rules_from:
        return _rules_for_sheet(repo_root, str(rules_from), sheet)
    return {}


def _phase_config_overrides(manifest: dict[str, Any], sheet: str) -> dict[str, Any]:
    overrides = dict(manifest.get("overrides", {}) or {})
    reserved = {
        "seeds.default",
        "outputs.root_template",
        "transfer_weights.per_dataset_multiplier",
        "transfer_weights.per_dataset_multiplier_from",
        "sheet_overrides",
    }
    out = {str(k): v for k, v in overrides.items() if str(k) not in reserved}
    sheet_overrides = overrides.get("sheet_overrides", {})
    if isinstance(sheet_overrides, dict):
        cur = sheet_overrides.get(sheet, {})
        if isinstance(cur, dict):
            out.update({str(k): v for k, v in cur.items()})
    return out


def _generated_config(
    *,
    repo_root: Path,
    manifest: dict[str, Any],
    sheet: str,
) -> tuple[Path, dict[str, Any]]:
    phase = str(manifest["phase"])
    base_configs = dict(manifest.get("base_configs", {}) or {})
    if sheet not in base_configs:
        raise ValueError(f"Manifest {phase} is missing base config for {sheet}")
    base_path = repo_root / str(base_configs[sheet])
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    raw = _load_yaml(base_path)
    overrides = dict(manifest.get("overrides", {}) or {})
    seed_expr = str(overrides.get("seeds.default", ""))
    if not seed_expr:
        raise ValueError(f"Manifest {phase} is missing overrides.seeds.default")
    expected_seeds = _load_seed_band(phase, repo_root)
    actual_seeds = _parse_seeds(seed_expr)
    if actual_seeds != expected_seeds:
        raise ValueError(f"{phase} seed band mismatch: manifest={actual_seeds}, expected={expected_seeds}")
    if bool(manifest.get("guardrails", {}).get("no_seed_0_9", False)) and any(0 <= seed <= 9 for seed in actual_seeds):
        raise ValueError(f"{phase} is not allowed to use final seeds 0..9")

    root_template = str(overrides.get("outputs.root_template", ""))
    if not root_template:
        raise ValueError(f"Manifest {phase} is missing outputs.root_template")

    raw.setdefault("seeds", {})["default"] = seed_expr
    raw.setdefault("outputs", {})["root"] = root_template.format(sheet=sheet)
    raw.setdefault("transfer_weights", {})["per_dataset_multiplier"] = _phase_weights(repo_root, manifest, sheet)
    for dotted_path, value in _phase_config_overrides(manifest, sheet).items():
        _apply_dotted_override(raw, dotted_path, value)

    out_dir = repo_root / str(manifest.get("outputs", {}).get("generated_config_dir", "v4/reports/_generated_configs"))
    out_path = out_dir / f"{phase}_{sheet}.yaml"
    return out_path, raw


def run_phase(repo_root: Path, phase: str, generate_config_only: bool) -> list[Path]:
    if phase not in PHASE_MANIFESTS:
        raise ValueError(f"Unknown phase {phase}; expected one of {sorted(PHASE_MANIFESTS)}")
    manifest_path = repo_root / PHASE_MANIFESTS[phase]
    manifest = _load_yaml(manifest_path)
    if str(manifest.get("phase", "")) != phase:
        raise ValueError(f"Manifest phase mismatch in {manifest_path}")

    _require_paths(repo_root, [str(p) for p in manifest.get("requires", []) or []])
    generated_paths: list[Path] = []
    for sheet in ("S4", "S5"):
        out_path, raw = _generated_config(repo_root=repo_root, manifest=manifest, sheet=sheet)
        _write_yaml(out_path, raw)
        generated_paths.append(out_path)
        print(f"wrote {out_path}")

    if generate_config_only:
        return generated_paths

    for path in generated_paths:
        cmd = [sys.executable, "-m", "rtbench.run", "--config", str(path), "--no-download"]
        print("running " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=repo_root, check=True)

    stamp = repo_root / str(manifest.get("outputs", {}).get("stamp", f"v4/reports/_stamps/{phase}.ok"))
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text("ok\n", encoding="utf-8")
    print(f"wrote {stamp}")
    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a locked RTBench v4 phase")
    parser.add_argument("--phase", required=True, choices=sorted(PHASE_MANIFESTS))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--generate-config-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_phase(repo_root, args.phase, bool(args.generate_config_only))


if __name__ == "__main__":
    main()

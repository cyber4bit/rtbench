from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


BASE_CONFIGS = {
    "S4": Path(".claude/worktrees/romantic-wilson-87e23a/outputs_v3_s4/config.resolved.yaml"),
    "S5": Path(".claude/worktrees/romantic-wilson-87e23a/outputs_v3_s5/config.resolved.yaml"),
}
MODES = {"S4": "RPLC", "S5": "HILIC"}


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


def _parse_seeds(expr: str) -> list[int]:
    text = str(expr).strip()
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _complete(run_dir: Path, seeds: list[int], expected_dataset_count: int) -> bool:
    per_seed = run_dir / "metrics" / "per_seed.csv"
    if not per_seed.exists():
        return False
    try:
        df = pd.read_csv(per_seed, dtype={"dataset": str}, encoding="utf-8")
    except Exception:
        return False
    if "dataset" not in df.columns or "seed" not in df.columns:
        return False
    expected = set(int(s) for s in seeds)
    counts = df.groupby(df["dataset"].astype(str).str.zfill(4))["seed"].apply(lambda s: set(int(x) for x in s))
    return bool(len(counts) == int(expected_dataset_count) and all(expected.issubset(v) for v in counts))


def _write_vs_unirt(run_dir: Path, baseline_csv: Path, mode: str, out_path: Path) -> None:
    per_seed = pd.read_csv(run_dir / "metrics" / "per_seed.csv", dtype={"dataset": str}, encoding="utf-8")
    per_seed["dataset"] = per_seed["dataset"].astype(str).str.zfill(4)
    base = pd.read_csv(baseline_csv, dtype={"dataset": str}, encoding="utf-8")
    base.columns = [str(c).strip().lower() for c in base.columns]
    base["dataset"] = base["dataset"].astype(str).str.zfill(4)
    if "mode" in base.columns:
        base = base.loc[base["mode"].astype(str).str.upper() == mode.upper()].copy()
    base = base.loc[base["method"].astype(str).str.lower() == "uni-rt"].copy()
    base = base.set_index("dataset")

    rows: list[dict[str, Any]] = []
    for dataset, cur in per_seed.groupby("dataset", sort=True):
        if dataset not in base.index:
            continue
        b = base.loc[dataset]
        our_mae = float(cur["mae"].mean())
        our_r2 = float(cur["r2"].mean())
        uni_mae = float(b["mae"])
        uni_r2 = float(b["r2"])
        beat_mae = bool(our_mae < uni_mae)
        beat_r2 = bool(our_r2 > uni_r2)
        rows.append(
            {
                "dataset": dataset,
                "our_mae_mean_10s": our_mae,
                "our_r2_mean_10s": our_r2,
                "uni_rt_mae": uni_mae,
                "uni_rt_r2": uni_r2,
                "delta_mae": uni_mae - our_mae,
                "delta_r2": our_r2 - uni_r2,
                "beat_mae": beat_mae,
                "beat_r2": beat_r2,
                "beat_both": bool(beat_mae and beat_r2),
                "seed_count": int(cur["seed"].nunique()),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def run(repo_root: Path, seeds_expr: str, out_root: Path, force: bool) -> None:
    seeds = _parse_seeds(seeds_expr)
    if any(0 <= seed <= 9 for seed in seeds):
        raise ValueError("v3 tuning must not use final seeds 0..9")
    gen_dir = repo_root / "v4/reports/_generated_configs"
    for sheet, base_rel in BASE_CONFIGS.items():
        base_path = repo_root / base_rel
        raw = _load_yaml(base_path)
        run_dir = out_root / f"outputs_v3_{sheet.lower()}"
        expected_dataset_count = len(raw.get("datasets", {}).get("external", []) or [])
        raw.setdefault("seeds", {})["default"] = seeds_expr
        raw.setdefault("outputs", {})["root"] = run_dir.as_posix()
        raw.setdefault("outputs", {})["resume"] = True
        cfg_path = gen_dir / f"v3_tuning_{sheet}.yaml"
        _write_yaml(cfg_path, raw)
        if not force and _complete(repo_root / run_dir, seeds, expected_dataset_count):
            print(f"skip complete {run_dir}")
        else:
            cmd = [sys.executable, "-m", "rtbench.run", "--config", str(cfg_path), "--no-download"]
            print("running " + " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=repo_root, check=True)
        _write_vs_unirt(
            repo_root / run_dir,
            repo_root / "data/baseline/unirt_sota_28.csv",
            MODES[sheet],
            repo_root / run_dir / "v3_vs_uni_rt.csv",
        )
        print(f"wrote {run_dir / 'v3_vs_uni_rt.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen v3 configs on v4 L2 tuning seeds.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--seeds", default="23:25")
    parser.add_argument("--out-root", default="v4/reports/_v3_tuning")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_root = Path(args.out_root)
    run(repo_root, args.seeds, out_root, bool(args.force))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import importlib
import inspect
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_REPO_ROOT_TEXT = str(SCRIPT_REPO_ROOT)
if SCRIPT_REPO_ROOT_TEXT not in sys.path:
    sys.path.insert(0, SCRIPT_REPO_ROOT_TEXT)

from rtbench.report_vs_unirt import write_unirt_report


DEFAULT_BASELINE = Path("data/baseline/unirt_sota_28.csv")
CORE_MODULE = "rtbench.bench.unified_cv"
CORE_RUNNER_NAMES = ("run_unified_cv", "run_strict_unified_cv", "run")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strict v7 unified CV and write Uni-RT-aligned reports.")
    parser.add_argument("--sheet", required=True, choices=("S4", "S5", "s4", "s5"))
    parser.add_argument("--mode", required=True, choices=("RPLC", "HILIC", "rplc", "hilic"))
    parser.add_argument("--config", required=True, help="v7 unified CV config or phase manifest")
    parser.add_argument("--folds", type=int, default=10, help="Number of strict CV folds")
    parser.add_argument("--shuffle-seed", type=int, default=20260505, help="Seed used to shuffle fold assignment")
    parser.add_argument("--output-root", required=True, help="Output directory for metrics and reports")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="Uni-RT baseline CSV")
    parser.add_argument("--repo-root", default="", help="Repository root; defaults to the script's repository")
    parser.add_argument("--no-download", action="store_true", help="Do not download or refresh external data")
    return parser


def _load_core_runner() -> Callable[..., Any]:
    try:
        module = importlib.import_module(CORE_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == CORE_MODULE:
            raise RuntimeError(
                f"Core unified CV module '{CORE_MODULE}' is not available in this branch. "
                "Merge with the core worker that provides rtbench/bench/unified_cv.py, then rerun this CLI."
            ) from exc
        raise

    for name in CORE_RUNNER_NAMES:
        runner = getattr(module, name, None)
        if callable(runner):
            return runner
    raise RuntimeError(
        f"Core module '{CORE_MODULE}' does not expose one of: {', '.join(CORE_RUNNER_NAMES)}"
    )


def _resolve_config_path(repo_root: Path, config_arg: str, sheet: str) -> Path:
    path = Path(config_arg)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        return path

    is_phase_manifest = str(raw.get("kind", "")).strip().lower() == "phase_manifest" or "base_configs" in raw
    if not is_phase_manifest:
        return path

    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    from v7.scripts.run_phase import PHASE_ALIASES, _generated_config, _merge_manifest, _write_yaml

    phase = str(raw.get("phase", "V7_validate"))
    manifest = _merge_manifest(raw, PHASE_ALIASES.get(phase), phase)
    generated_path, generated_raw = _generated_config(repo_root, manifest, sheet.upper())
    _write_yaml(generated_path, generated_raw)
    return generated_path.resolve()


def _call_core_runner(
    runner: Callable[..., Any],
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    config_path: Path,
) -> Any:
    values = {
        "sheet": args.sheet.upper(),
        "mode": args.mode.upper(),
        "config_path": config_path,
        "folds": int(args.folds),
        "n_folds": int(args.folds),
        "shuffle_seed": int(args.shuffle_seed),
        "output_root": output_root,
        "output_dir": output_root,
        "repo_root": repo_root,
        "no_download": bool(args.no_download),
        "download": not bool(args.no_download),
    }

    sig = inspect.signature(runner)
    if "config_path" not in sig.parameters and "config" in sig.parameters:
        values["config"] = str(config_path)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return runner(**values)

    kwargs = {name: values[name] for name in sig.parameters if name in values}
    if kwargs:
        return runner(**kwargs)
    return runner()


def _write_per_seed_from_result(result: Any, per_seed_path: Path) -> Path:
    metrics_dir = per_seed_path.parent
    metrics_dir.mkdir(parents=True, exist_ok=True)

    candidate = result
    if isinstance(result, dict):
        for key in ("per_seed", "per_seed_metrics", "per_seed_df", "per_seed_path", "metrics_path"):
            if key in result:
                candidate = result[key]
                break

    if isinstance(candidate, pd.DataFrame):
        candidate.to_csv(per_seed_path, index=False, encoding="utf-8")
        return per_seed_path

    if isinstance(candidate, (str, Path)):
        src = Path(candidate)
        if not src.exists():
            raise FileNotFoundError(f"Core returned per-seed path that does not exist: {src}")
        if src.resolve() != per_seed_path.resolve():
            shutil.copyfile(src, per_seed_path)
        return per_seed_path

    if per_seed_path.exists():
        return per_seed_path

    raise RuntimeError(
        "Core unified CV completed but did not return or write metrics/per_seed.csv. "
        "Expected a DataFrame, a path, a dict containing per_seed/per_seed_path, or the file on disk."
    )


def run(args: argparse.Namespace) -> dict[str, Path]:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_script()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if int(args.folds) <= 0:
        raise ValueError("--folds must be positive")

    config_path = _resolve_config_path(repo_root, args.config, args.sheet)
    runner = _load_core_runner()
    result = _call_core_runner(runner, args, repo_root, output_root, config_path)

    metrics_dir = output_root / "metrics"
    per_seed_path = _write_per_seed_from_result(result, metrics_dir / "per_seed.csv")
    baseline = Path(args.baseline)
    if not baseline.is_absolute():
        baseline = repo_root / baseline

    report_path = output_root / "report_vs_unirt.md"
    write_unirt_report(
        out_path=report_path,
        per_seed=per_seed_path,
        baseline_csv=baseline,
        mode=args.mode.upper(),
        ours_label="RTBench strict unified CV",
        n_model=1,
        expected_seeds=list(range(int(args.folds))),
        output_dir=metrics_dir,
    )

    return {
        "output_root": output_root,
        "per_seed": per_seed_path,
        "report": report_path,
        "cross_summary": metrics_dir / "cross_dataset_summary_vs_unirt.csv",
        "dataset_comparison": metrics_dir / "dataset_level_comparison_vs_unirt.csv",
        "significance": metrics_dir / "significance_vs_unirt.csv",
        "per_dataset_stats": metrics_dir / "per_dataset_10run_stats.csv",
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = run(args)
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()

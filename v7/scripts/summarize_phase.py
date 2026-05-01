from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd


THRESHOLDS = {
    "S4": {"avg_mae_lt": 25.5482, "avg_r2_gt": 0.9144, "beat_both_min": 10, "mode": "RPLC"},
    "S5": {"avg_mae_lt": 48.0916, "avg_r2_gt": 0.8305, "beat_both_min": 10, "mode": "HILIC"},
}


def _slug(phase: str) -> str:
    text = str(phase).strip()
    if text.startswith("V7_"):
        text = text[3:]
    return text.lower()


def _phase_dir(phase: str, sheet: str) -> Path:
    return Path(f"outputs_v7_{_slug(phase)}_{sheet}")


def _baseline(repo_root: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_csv(repo_root / "data/baseline/unirt_sota_28.csv", dtype={"dataset": str}, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["dataset"] = df["dataset"].astype(str).str.zfill(4)
    df = df.loc[df["mode"].astype(str).str.upper() == THRESHOLDS[sheet]["mode"]].copy()
    df = df.loc[df["method"].astype(str).str.lower() == "uni-rt"].copy()
    return df.set_index("dataset")


def _comparison(repo_root: Path, phase: str, sheet: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = repo_root / _phase_dir(phase, sheet)
    per_seed_path = run_dir / "metrics/per_seed.csv"
    if not per_seed_path.exists():
        raise FileNotFoundError(f"Missing per-seed metrics: {per_seed_path}")
    per_seed = pd.read_csv(per_seed_path, dtype={"dataset": str}, encoding="utf-8")
    per_seed["dataset"] = per_seed["dataset"].astype(str).str.zfill(4)
    per_seed["seed"] = per_seed["seed"].astype(int)
    base = _baseline(repo_root, sheet)
    rows: list[dict[str, Any]] = []
    for dataset, cur in per_seed.groupby("dataset", sort=True):
        if dataset not in base.index:
            continue
        b = base.loc[dataset]
        our_mae = float(cur["mae"].mean())
        our_r2 = float(cur["r2"].mean())
        uni_mae = float(b["mae"])
        uni_r2 = float(b["r2"])
        rows.append(
            {
                "phase": phase,
                "sheet": sheet,
                "dataset": dataset,
                "our_mae_mean": our_mae,
                "our_r2_mean": our_r2,
                "uni_rt_mae": uni_mae,
                "uni_rt_r2": uni_r2,
                "delta_mae": uni_mae - our_mae,
                "delta_r2": our_r2 - uni_r2,
                "beat_mae": bool(our_mae < uni_mae),
                "beat_r2": bool(our_r2 > uni_r2),
                "beat_both": bool(our_mae < uni_mae and our_r2 > uni_r2),
                "seed_count": int(cur["seed"].nunique()),
            }
        )
    comp = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    out_path = run_dir / "metrics/vs_unirt.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_path, index=False, encoding="utf-8")
    summary = {
        "phase": phase,
        "sheet": sheet,
        "avg_mae": float(comp["our_mae_mean"].mean()) if not comp.empty else float("inf"),
        "avg_r2": float(comp["our_r2_mean"].mean()) if not comp.empty else float("-inf"),
        "beat_both": int(comp["beat_both"].sum()) if not comp.empty else 0,
        "dataset_count": int(comp.shape[0]),
        "seed_counts": sorted(int(x) for x in comp["seed_count"].unique()) if not comp.empty else [],
    }
    return comp, summary


def _failure_table(comp: pd.DataFrame) -> pd.DataFrame:
    if comp.empty:
        return pd.DataFrame()
    failed = comp.loc[~comp["beat_both"].astype(bool)].copy()
    if failed.empty:
        return failed
    failed["mae_gap"] = (failed["our_mae_mean"] - failed["uni_rt_mae"]).clip(lower=0.0)
    failed["r2_gap"] = (failed["uni_rt_r2"] - failed["our_r2_mean"]).clip(lower=0.0)
    failed["priority_score"] = failed["mae_gap"] + (100.0 * failed["r2_gap"])
    failed["failure_reason"] = failed.apply(
        lambda row: ",".join(
            name
            for name, missed in (
                ("mae", not bool(row["beat_mae"])),
                ("r2", not bool(row["beat_r2"])),
            )
            if missed
        ),
        axis=1,
    )
    cols = [
        "phase",
        "sheet",
        "dataset",
        "failure_reason",
        "priority_score",
        "mae_gap",
        "r2_gap",
        "our_mae_mean",
        "uni_rt_mae",
        "our_r2_mean",
        "uni_rt_r2",
        "delta_mae",
        "delta_r2",
        "seed_count",
    ]
    return failed[cols].sort_values(["sheet", "priority_score", "dataset"], ascending=[True, False, True])


def _candidate_summary(repo_root: Path, phase: str, sheet: str) -> pd.DataFrame:
    path = repo_root / _phase_dir(phase, sheet) / "metrics/candidate_diagnostics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"dataset": str}, encoding="utf-8")
    if df.empty or "candidate_name" not in df.columns:
        return pd.DataFrame()
    df["dataset"] = df["dataset"].astype(str).str.zfill(4)
    df["sheet"] = sheet
    group_cols = ["sheet", "dataset", "candidate_name"]
    agg: dict[str, Any] = {
        "selected": "sum",
        "candidate_rank": "count",
    }
    for col in ("val_mae", "val_r2", "test_mae", "test_r2", "weight"):
        if col in df.columns:
            agg[col] = "mean"
    out = df.groupby(group_cols, dropna=False).agg(agg).reset_index()
    out = out.rename(columns={"selected": "selected_count", "candidate_rank": "observed_count"})
    return out.sort_values(["sheet", "dataset", "selected_count", "candidate_name"], ascending=[True, True, False, True])


def summarize(repo_root: Path, phase: str) -> None:
    report_dir = repo_root / "v7/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    checks = []
    failure_tables = []
    candidate_tables = []
    for sheet in ("S4", "S5"):
        comp, summary = _comparison(repo_root, phase, sheet)
        failure_tables.append(_failure_table(comp))
        candidate_tables.append(_candidate_summary(repo_root, phase, sheet))
        summaries.append(summary)
        threshold = THRESHOLDS[sheet]
        checks.extend(
            [
                (sheet, "avg_mae", summary["avg_mae"] < threshold["avg_mae_lt"], f"{summary['avg_mae']:.4f} < {threshold['avg_mae_lt']}"),
                (sheet, "avg_r2", summary["avg_r2"] > threshold["avg_r2_gt"], f"{summary['avg_r2']:.4f} > {threshold['avg_r2_gt']}"),
                (sheet, "beat_both", summary["beat_both"] >= threshold["beat_both_min"], f"{summary['beat_both']} >= {threshold['beat_both_min']}"),
            ]
        )

    summary_path = report_dir / f"{phase}_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    failures = pd.concat([df for df in failure_tables if not df.empty], ignore_index=True) if any(not df.empty for df in failure_tables) else pd.DataFrame()
    failures_path = report_dir / f"{phase}_failures.csv"
    if not failures.empty:
        failures.to_csv(failures_path, index=False, encoding="utf-8")
    else:
        failures_path.write_text("", encoding="utf-8")

    candidates = pd.concat([df for df in candidate_tables if not df.empty], ignore_index=True) if any(not df.empty for df in candidate_tables) else pd.DataFrame()
    candidates_path = report_dir / f"{phase}_candidate_summary.csv"
    if not candidates.empty:
        candidates.to_csv(candidates_path, index=False, encoding="utf-8")
    else:
        candidates_path.write_text("", encoding="utf-8")

    ok = all(item[2] for item in checks)
    lines = [
        f"# RTBench v7 {phase} Report",
        "",
        f"- Status: {'PASS' if ok else 'FAIL'}",
        "- Model constraint: sheet-level unified Hyper-TL only; lookup, HILIC pool, local fast, and per-dataset overrides disabled.",
        "",
        "| Sheet | Avg MAE | Avg R2 | Beat Both | Dataset Count | Seed Counts |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['sheet']} | {summary['avg_mae']:.4f} | {summary['avg_r2']:.4f} | "
            f"{summary['beat_both']} | {summary['dataset_count']} | {summary['seed_counts']} |"
        )
    lines.extend(["", "| Sheet | Check | Pass | Detail |", "| --- | --- | --- | --- |"])
    for sheet, name, passed, detail in checks:
        lines.append(f"| {sheet} | {name} | {'PASS' if passed else 'FAIL'} | {detail} |")
    lines.extend(["", "## Failure Priorities", ""])
    if failures.empty:
        lines.append("No failed beat-both datasets.")
    else:
        lines.extend(
            [
                "| Sheet | Dataset | Reason | Priority | MAE Gap | R2 Gap | Our MAE | Uni-RT MAE | Our R2 | Uni-RT R2 |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in failures.sort_values(["sheet", "priority_score", "dataset"], ascending=[True, False, True]).to_dict("records"):
            lines.append(
                f"| {row['sheet']} | {row['dataset']} | {row['failure_reason']} | "
                f"{row['priority_score']:.4f} | {row['mae_gap']:.4f} | {row['r2_gap']:.4f} | "
                f"{row['our_mae_mean']:.4f} | {row['uni_rt_mae']:.4f} | "
                f"{row['our_r2_mean']:.4f} | {row['uni_rt_r2']:.4f} |"
            )
    lines.extend(["", "## Candidate Diagnostics", ""])
    if candidates.empty:
        lines.append("No candidate diagnostics were found for this phase.")
    else:
        lines.append(f"Candidate summary written to `{candidates_path.as_posix()}`.")
    report_path = report_dir / f"{phase}_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"wrote {failures_path}")
    print(f"wrote {candidates_path}")
    print(f"wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a v7 phase against Uni-RT baselines.")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    summarize(Path(args.repo_root).resolve(), str(args.phase))


if __name__ == "__main__":
    main()

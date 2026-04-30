from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from verify_cp_injection import scan


THRESHOLDS = {
    "S4": {"avg_mae_lt": 25.5482, "avg_r2_gt": 0.9144, "beat_both_min": 10, "mode": "RPLC"},
    "S5": {"avg_mae_lt": 48.0916, "avg_r2_gt": 0.8305, "beat_both_min": 10, "mode": "HILIC"},
}
EXPECTED_SEEDS = {
    "V6_validate": list(range(30, 40)),
    "V6_final": list(range(40, 50)),
    "V6_reseal_validate": list(range(50, 60)),
    "V6_reseal_final": list(range(60, 70)),
}
BASE_CONFIGS = {
    "S4": Path("v4/reports/_v3_tuning/outputs_v3_s4/config.resolved.yaml"),
    "S5": Path("v4/reports/_v3_tuning/outputs_v3_s5/config.resolved.yaml"),
}
PHASE_OUTPUT_SLUGS = {
    "V6_validate": "validate",
    "V6_final": "final",
    "V6_reseal_validate": "reseal_validate",
    "V6_reseal_final": "reseal_final",
}
GATE_REPORTS = {
    "V6_validate": {
        "title": "v6 Validate Gate Report",
        "report": Path("v6/reports/gate_report.md"),
        "summary": Path("v6/reports/v6_summary.csv"),
        "stamp": Path("v6/reports/_stamps/gate.ok"),
    },
    "V6_reseal_validate": {
        "title": "v6 Reseal Validate Gate Report",
        "report": Path("v6/reports/reseal_gate_report.md"),
        "summary": Path("v6/reports/v6_reseal_summary.csv"),
        "stamp": Path("v6/reports/_stamps/reseal_gate.ok"),
    },
}
FINAL_REPORTS = {
    "V6_final": {
        "title": "v6 Final Gate Report",
        "report_title": "RTBench v6 Final Report",
        "requires": Path("v6/reports/_stamps/gate.ok"),
        "gate_report": Path("v6/reports/final_gate_report.md"),
        "summary": Path("v6/reports/v6_summary.csv"),
        "report": Path("v6/reports/v6_report.md"),
        "stamp": Path("v6/reports/_stamps/report.ok"),
    },
    "V6_reseal_final": {
        "title": "v6 Reseal Final Gate Report",
        "report_title": "RTBench v6 Reseal Final Report",
        "requires": Path("v6/reports/_stamps/reseal_gate.ok"),
        "gate_report": Path("v6/reports/reseal_final_gate_report.md"),
        "summary": Path("v6/reports/v6_reseal_summary.csv"),
        "report": Path("v6/reports/v6_reseal_report.md"),
        "stamp": Path("v6/reports/_stamps/reseal_report.ok"),
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, dotted))
        else:
            out[dotted] = value
    return out


def _allowed(path: str, allowed_keys: set[str]) -> bool:
    return any(path == key or path.startswith(key + ".") for key in allowed_keys)


def _config_diff_findings(repo_root: Path, phase: str, sheet: str) -> list[str]:
    whitelist = _load_yaml(repo_root / "v6/configs/config_diff_whitelist.yaml")
    allowed_keys = set(str(x) for x in whitelist.get("allowed_keys", []))
    base = _flatten(_load_yaml(repo_root / BASE_CONFIGS[sheet]))
    generated = _flatten(_load_yaml(repo_root / f"v6/reports/_generated_configs/{phase}_{sheet}.yaml"))
    findings: list[str] = []
    for key in sorted(set(base) | set(generated)):
        if base.get(key) != generated.get(key) and not _allowed(key, allowed_keys):
            findings.append(key)
    return findings


def _baseline(repo_root: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_csv(repo_root / "data/baseline/unirt_sota_28.csv", dtype={"dataset": str}, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["dataset"] = df["dataset"].astype(str).str.zfill(4)
    df = df.loc[df["mode"].astype(str).str.upper() == THRESHOLDS[sheet]["mode"]].copy()
    df = df.loc[df["method"].astype(str).str.lower() == "uni-rt"].copy()
    return df.set_index("dataset")


def _phase_dir(phase: str, sheet: str) -> Path:
    return Path(f"outputs_v6_{PHASE_OUTPUT_SLUGS[phase]}_{sheet}")


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
        beat_mae = our_mae < uni_mae
        beat_r2 = our_r2 > uni_r2
        rows.append(
            {
                "dataset": dataset,
                "our_mae_mean": our_mae,
                "our_r2_mean": our_r2,
                "uni_rt_mae": uni_mae,
                "uni_rt_r2": uni_r2,
                "delta_mae": uni_mae - our_mae,
                "delta_r2": our_r2 - uni_r2,
                "beat_mae": bool(beat_mae),
                "beat_r2": bool(beat_r2),
                "beat_both": bool(beat_mae and beat_r2),
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


def _seed_findings(repo_root: Path, phase: str, sheet: str) -> list[str]:
    expected = set(EXPECTED_SEEDS[phase])
    run_dir = repo_root / _phase_dir(phase, sheet)
    per_seed = pd.read_csv(run_dir / "metrics/per_seed.csv", dtype={"dataset": str}, encoding="utf-8")
    per_seed["dataset"] = per_seed["dataset"].astype(str).str.zfill(4)
    per_seed["seed"] = per_seed["seed"].astype(int)
    findings = []
    for dataset, cur in per_seed.groupby("dataset", sort=True):
        got = set(int(x) for x in cur["seed"].tolist())
        if got != expected:
            findings.append(f"{sheet}/{dataset} seeds={sorted(got)} expected={sorted(expected)}")
    return findings


def _cp_findings(repo_root: Path) -> list[str]:
    findings = []
    static = scan(repo_root / "rtbench")
    if static:
        findings.append(f"static CP guard failed with {len(static)} finding(s)")
    audit_path = repo_root / "v6/reports/cp_audit.json"
    if not audit_path.exists():
        findings.append("cp_audit.json missing")
        return findings
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if bool(audit.get("hard_bug", False)):
        findings.append("cp_audit hard_bug=true")
    grad = audit.get("film_grad_ok", "n/a")
    if grad not in (True, "true", "n/a"):
        findings.append(f"cp_audit film_grad_ok={grad!r}")
    return findings


def _registry_findings(repo_root: Path, phase: str, sheet: str) -> list[str]:
    registry_path = repo_root / "experiments/registry.csv"
    run_dir = _phase_dir(phase, sheet).as_posix()
    sha_path = repo_root / _phase_dir(phase, sheet) / "config.sha1"
    if not registry_path.exists() or not sha_path.exists():
        return [f"{sheet} registry/config hash input missing"]
    reg = pd.read_csv(registry_path, dtype=str, encoding="utf-8").fillna("")
    rows = reg.loc[reg["run_dir"] == run_dir]
    if rows.empty:
        return [f"{sheet} registry row missing for {run_dir}"]
    sha = sha_path.read_text(encoding="utf-8").strip()
    if str(rows.iloc[-1].get("config_sha1", "")).strip() != sha:
        return [f"{sheet} registry config_sha1 mismatch"]
    return []


def _gate_rows(repo_root: Path, phase: str, require_registry: bool) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    all_pass = True
    cp_errors = _cp_findings(repo_root)
    for sheet in ("S4", "S5"):
        _, summary = _comparison(repo_root, phase, sheet)
        threshold = THRESHOLDS[sheet]
        checks = [
            ("avg_mae", summary["avg_mae"] < float(threshold["avg_mae_lt"]), f"{summary['avg_mae']:.4f} < {threshold['avg_mae_lt']}"),
            ("avg_r2", summary["avg_r2"] > float(threshold["avg_r2_gt"]), f"{summary['avg_r2']:.4f} > {threshold['avg_r2_gt']}"),
            ("beat_both", summary["beat_both"] >= int(threshold["beat_both_min"]), f"{summary['beat_both']} >= {threshold['beat_both_min']}"),
            ("seed_count", not _seed_findings(repo_root, phase, sheet), "exact expected seed set"),
            ("config_diff", not _config_diff_findings(repo_root, phase, sheet), "within whitelist"),
        ]
        if require_registry:
            checks.append(("registry", not _registry_findings(repo_root, phase, sheet), "registry config hash matches"))
        for name, ok, detail in checks:
            rows.append({"phase": phase, "sheet": sheet, "check": name, "pass": bool(ok), "detail": detail})
            all_pass = all_pass and bool(ok)
    ok_cp = not cp_errors
    rows.append({"phase": phase, "sheet": "ALL", "check": "cp_guard_and_audit", "pass": ok_cp, "detail": "; ".join(cp_errors) if cp_errors else "PASS"})
    all_pass = all_pass and ok_cp
    return rows, all_pass


def _write_gate_report(repo_root: Path, rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    lines = [f"# {title}", "", "| Phase | Sheet | Check | Pass | Detail |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['sheet']} | {row['check']} | {'PASS' if row['pass'] else 'FAIL'} | {row['detail']} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_csv(repo_root: Path, phase: str, out_path: Path) -> None:
    rows = []
    for sheet in ("S4", "S5"):
        _, summary = _comparison(repo_root, phase, sheet)
        rows.append(summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gate_check(repo_root: Path, phase: str = "V6_validate") -> None:
    if phase not in GATE_REPORTS:
        raise SystemExit(f"{phase} is not a validate gate phase")
    meta = GATE_REPORTS[phase]
    rows, ok = _gate_rows(repo_root, phase, require_registry=False)
    _write_gate_report(repo_root, rows, repo_root / meta["report"], str(meta["title"]))
    _write_summary_csv(repo_root, phase, repo_root / meta["summary"])
    if ok:
        stamp = repo_root / meta["stamp"]
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text("ok\n", encoding="utf-8")
        print(f"wrote {stamp}")
    else:
        raise SystemExit(f"{phase} gate failed; not creating {meta['stamp']}")


def _sha(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def _v3_hashes(repo_root: Path) -> dict[str, str]:
    return {
        "S4": _sha(repo_root / "v4/reports/_v3_tuning/outputs_v3_s4/config.sha1"),
        "S5": _sha(repo_root / "v4/reports/_v3_tuning/outputs_v3_s5/config.sha1"),
    }


def emit_report(repo_root: Path, phase: str = "V6_final") -> None:
    if phase not in FINAL_REPORTS:
        raise SystemExit(f"{phase} is not a final report phase")
    meta = FINAL_REPORTS[phase]
    if not (repo_root / meta["requires"]).exists():
        raise SystemExit(f"Refusing final report before {meta['requires']}")
    rows, ok = _gate_rows(repo_root, phase, require_registry=True)
    _write_gate_report(repo_root, rows, repo_root / meta["gate_report"], str(meta["title"]))
    _write_summary_csv(repo_root, phase, repo_root / meta["summary"])
    summaries = []
    for sheet in ("S4", "S5"):
        _, summary = _comparison(repo_root, phase, sheet)
        summaries.append(summary)
    report = [
        f"# {meta['report_title']}",
        "",
        f"- Status: {'PASS' if ok else 'FAIL'}",
        f"- v3 config hashes: S4 `{_v3_hashes(repo_root)['S4']}`, S5 `{_v3_hashes(repo_root)['S5']}`",
        f"- v6 config hashes: S4 `{_sha(repo_root / (_phase_dir(phase, 'S4') / 'config.sha1'))}`, S5 `{_sha(repo_root / (_phase_dir(phase, 'S5') / 'config.sha1'))}`",
        "",
        "| Sheet | Avg MAE | Avg R2 | Beat Both | Dataset Count | Seed Counts |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        report.append(
            f"| {summary['sheet']} | {summary['avg_mae']:.4f} | {summary['avg_r2']:.4f} | "
            f"{summary['beat_both']} | {summary['dataset_count']} | {summary['seed_counts']} |"
        )
    report.extend(["", f"See `{meta['gate_report']}` for binary gate checks."])
    out = repo_root / meta["report"]
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    if ok:
        stamp = repo_root / meta["stamp"]
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text("ok\n", encoding="utf-8")
        print(f"wrote {stamp}")
    else:
        raise SystemExit(f"{phase} final gate failed; report is diagnostic only")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize v6 gates and reports.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gate-check", action="store_true")
    group.add_argument("--emit-report", action="store_true")
    parser.add_argument("--phase", default=None, choices=sorted(set(GATE_REPORTS) | set(FINAL_REPORTS)))
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    if args.gate_check:
        gate_check(repo_root, args.phase or "V6_validate")
    else:
        emit_report(repo_root, args.phase or "V6_final")


if __name__ == "__main__":
    main()

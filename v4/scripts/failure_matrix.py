from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_V3_WORKTREE = Path(".claude") / "worktrees" / "romantic-wilson-87e23a"


def _read_rows(path: Path, sheet: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            row["sheet"] = sheet
            out.append(row)
        return out


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def _failure_kind(row: dict[str, str]) -> str:
    beat_mae = _bool(row, "beat_mae")
    beat_r2 = _bool(row, "beat_r2")
    if beat_mae and beat_r2:
        return "win"
    if beat_mae and not beat_r2:
        return "r2_only_loss"
    if not beat_mae and beat_r2:
        return "mae_only_loss"
    return "mae_and_r2_loss"


def _severity(row: dict[str, str]) -> str:
    dm = _float(row, "delta_mae")
    dr = _float(row, "delta_r2")
    if _bool(row, "beat_both"):
        return "win"
    if dm > -2.0 and dr > -0.01:
        return "near_miss"
    if dm < -25.0 or dr < -0.15:
        return "catastrophic"
    return "hard_loss"


def build_matrix(repo_root: Path, v3_root: Path) -> list[dict[str, object]]:
    rows = []
    rows.extend(_read_rows(v3_root / "outputs_v3_s4" / "v3_vs_uni_rt.csv", "S4"))
    rows.extend(_read_rows(v3_root / "outputs_v3_s5" / "v3_vs_uni_rt.csv", "S5"))
    matrix = []
    for row in rows:
        matrix.append(
            {
                "sheet": row["sheet"],
                "dataset": str(row.get("dataset", "")).zfill(4),
                "our_mae": _float(row, "our_mae_mean_10s"),
                "unirt_mae": _float(row, "uni_rt_mae"),
                "delta_mae": _float(row, "delta_mae"),
                "our_r2": _float(row, "our_r2_mean_10s"),
                "unirt_r2": _float(row, "uni_rt_r2"),
                "delta_r2": _float(row, "delta_r2"),
                "beat_mae": _bool(row, "beat_mae"),
                "beat_r2": _bool(row, "beat_r2"),
                "beat_both": _bool(row, "beat_both"),
                "failure_kind": _failure_kind(row),
                "severity": _severity(row),
            }
        )
    return matrix


def write_outputs(matrix: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "failure_matrix.csv"
    md_path = out_dir / "failure_matrix.md"
    fields = [
        "sheet",
        "dataset",
        "our_mae",
        "unirt_mae",
        "delta_mae",
        "our_r2",
        "unirt_r2",
        "delta_r2",
        "beat_mae",
        "beat_r2",
        "beat_both",
        "failure_kind",
        "severity",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matrix)

    failures = [r for r in matrix if not r["beat_both"]]
    lines = [
        "# v4 Failure Matrix",
        "",
        f"- Total rows: {len(matrix)}",
        f"- Failures: {len(failures)}",
        "",
        "| Sheet | Dataset | Kind | Severity | Delta MAE | Delta R2 |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for r in failures:
        lines.append(
            f"| {r['sheet']} | {r['dataset']} | {r['failure_kind']} | {r['severity']} | "
            f"{float(r['delta_mae']):.4f} | {float(r['delta_r2']):.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v4 failure matrix from v3 output CSVs.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--v3-root", default="", help="Path containing outputs_v3_s4/outputs_v3_s5.")
    parser.add_argument("--out-dir", default="v4/reports")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    v3_root = Path(args.v3_root).resolve() if args.v3_root else repo_root / DEFAULT_V3_WORKTREE
    matrix = build_matrix(repo_root=repo_root, v3_root=v3_root)
    if not matrix:
        raise SystemExit(f"No v3 rows found under {v3_root}")
    write_outputs(matrix, repo_root / args.out_dir)


if __name__ == "__main__":
    main()


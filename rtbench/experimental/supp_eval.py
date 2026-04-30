from __future__ import annotations

import argparse
import copy
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..bench import config_from_raw, parse_list_expr, prepare, run_trial
from ..config import parse_seed_range, resolve_config
from ..data import ensure_repo_data
from ..experiments import effective_config_file_sha1, record_experiment, status_for_run_dir, write_effective_config_snapshot
from ..logging_utils import attach_json_log, configure_logging, default_run_log_path


logger = logging.getLogger("rtbench.experimental.supp_eval")


def _safe_model_name(x: str) -> str:
    s = str(x).strip()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("/", "_").replace("-", "_")
    return s


def _is_target_row(v: Any) -> bool:
    s = str(v).strip()
    if not s:
        return False
    if s.lower() in ("avg", "std", "nan"):
        return False
    return bool(re.fullmatch(r"\d{3,4}", s))


def parse_supp_table(xlsx_path: Path, sheet_name: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    if df.shape[0] < 3 or df.shape[1] < 7:
        raise ValueError(f"Unexpected format in sheet '{sheet_name}'")

    top = df.iloc[0].tolist()
    metrics = df.iloc[1].tolist()
    model_cols: dict[str, dict[str, int]] = {}
    cur_model = ""
    for c in range(3, len(top)):
        if pd.notna(top[c]):
            cur_model = str(top[c]).strip()
        m = str(metrics[c]).strip() if pd.notna(metrics[c]) else ""
        if not cur_model or not m:
            continue
        model_cols.setdefault(cur_model, {})[m] = c

    rows = []
    for r in range(2, len(df)):
        ds = df.iloc[r, 0]
        if not _is_target_row(ds):
            continue
        ds_id = str(ds).zfill(4)
        rec = {"dataset": ds_id}
        for mname, mm in model_cols.items():
            mkey = _safe_model_name(mname)
            for metric in ("MAE", "R2"):
                col = mm.get(metric, None)
                if col is None:
                    continue
                rec[f"{mkey}_{metric.lower()}"] = float(df.iloc[r, col])
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No dataset rows parsed from '{sheet_name}'")
    model_names = [_safe_model_name(x) for x in model_cols.keys()]
    return out, model_names


def _rt_count_for_dataset(processed_root: Path, ds: str) -> int:
    can_p = processed_root / ds / f"{ds}_rtdata_canonical_success.tsv"
    iso_p = processed_root / ds / f"{ds}_rtdata_isomeric_success.tsv"
    can = pd.read_csv(can_p, sep="\t", encoding="utf-8") if can_p.exists() else pd.DataFrame(columns=["id"])
    iso = pd.read_csv(iso_p, sep="\t", encoding="utf-8") if iso_p.exists() else pd.DataFrame(columns=["id"])
    if "id" not in can.columns:
        can = pd.DataFrame(columns=["id"])
    if "id" not in iso.columns:
        iso = pd.DataFrame(columns=["id"])
    can = can.drop_duplicates(subset="id", keep="first")
    iso = iso.drop_duplicates(subset="id", keep="first")
    add = iso.loc[~iso["id"].isin(can["id"])]
    merged = pd.concat([can[["id"]], add[["id"]]], ignore_index=True)
    return int(merged["id"].nunique())


def _write_sheet_baseline(df_sheet: pd.DataFrame, out_csv: Path) -> None:
    # Use MDL-TL as paper baseline for run_trial compatibility.
    cand = [c for c in df_sheet.columns if c.endswith("_mae") and c.startswith("MDL_TL")]
    cand_r2 = [c for c in df_sheet.columns if c.endswith("_r2") and c.startswith("MDL_TL")]
    if not cand or not cand_r2:
        # fallback to first model with MAE/R2
        cand = [c for c in df_sheet.columns if c.endswith("_mae")]
        cand_r2 = [c for c in df_sheet.columns if c.endswith("_r2")]
    if not cand or not cand_r2:
        raise ValueError("No MAE/R2 baseline columns found in supplementary table")
    mae_col = cand[0]
    r2_col = cand_r2[0]
    b = df_sheet[["dataset", mae_col, r2_col]].copy()
    b.columns = ["dataset", "paper_mae", "paper_r2"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    b.to_csv(out_csv, index=False, encoding="utf-8")


def _sheet_to_key(sheet_name: str) -> str:
    s = sheet_name.strip().lower().replace(" ", "")
    if s in ("tables4", "s4"):
        return "S4"
    if s in ("tables5", "s5"):
        return "S5"
    return sheet_name.replace(" ", "_")


def run_sheet(
    base_raw: dict[str, Any],
    xlsx_path: Path,
    sheet_name: str,
    seeds: list[int],
    out_root: Path,
    no_download: bool,
    resume: bool = False,
    max_datasets: int | None = None,
    dataset_filter: set[str] | None = None,
) -> None:
    sheet_key = _sheet_to_key(sheet_name)
    sheet_all_df, model_names = parse_supp_table(xlsx_path=xlsx_path, sheet_name=sheet_name)
    run_df = sheet_all_df.copy()
    if dataset_filter:
        keep = {str(x).zfill(4) for x in dataset_filter}
        run_df = run_df.loc[run_df["dataset"].astype(str).str.zfill(4).isin(keep)].copy()
    if max_datasets is not None and max_datasets > 0:
        run_df = run_df.iloc[:max_datasets].copy()

    ds_pool_all = sheet_all_df["dataset"].tolist()
    ds_pool = run_df["dataset"].tolist()
    local_root = Path(base_raw["data"]["local_root"])
    processed_root = local_root / "processed_data"
    if not no_download:
        ensure_repo_data(
            repo_url=base_raw["data"]["repo_url"],
            commit=str(base_raw["data"]["commit"]),
            data_root=local_root,
            dataset_ids=ds_pool,
            download=True,
        )
    baseline_dir = Path(str(base_raw.get("data", {}).get("baseline_dir", "data/baseline")))
    baseline_csv = baseline_dir / f"supp_{sheet_key.lower()}_mdl_tl.csv"
    _write_sheet_baseline(sheet_all_df, baseline_csv)
    logger.info(
        "Running supplement sheet %s with %d target datasets.",
        sheet_key,
        len(ds_pool),
        extra={
            "sheet": sheet_key,
            "sheet_name": sheet_name,
            "target_dataset_count": len(ds_pool),
            "seed_count": len(seeds),
            "out_root": out_root.as_posix(),
        },
    )

    results: list[dict[str, Any]] = []
    for i, target_ds in enumerate(ds_pool, start=1):
        pretrain_ids = [d for d in ds_pool_all if d != target_ds]
        expected_cnt = sum(_rt_count_for_dataset(processed_root=processed_root, ds=d) for d in pretrain_ids)

        raw = copy.deepcopy(base_raw)
        raw["datasets"]["pretrain"] = pretrain_ids
        raw["datasets"]["external"] = [target_ds]
        raw["datasets"]["expected_pretrain_count"] = int(expected_cnt)
        raw["data"]["baseline_csv"] = str(baseline_csv).replace("\\", "/")
        raw["outputs"]["root"] = str((out_root / sheet_key / target_ds).as_posix())
        raw["outputs"]["resume"] = bool(resume)

        resolved_raw = copy.deepcopy(raw)
        cfg = config_from_raw(resolved_raw)
        sha1 = effective_config_file_sha1(resolved_raw)
        run_dir = Path(resolved_raw["outputs"]["root"])
        write_effective_config_snapshot(
            Path.cwd(),
            run_dir=run_dir,
            config_raw=resolved_raw,
            config_sha1=sha1,
            write_run_sha1=True,
        )
        with attach_json_log(default_run_log_path(run_dir)):
            logger.info(
                "[%s] [%d/%d] target_dataset=%s",
                sheet_key,
                i,
                len(ds_pool),
                target_ds,
                extra={
                    "sheet": sheet_key,
                    "target_dataset": target_ds,
                    "target_index": i,
                    "target_count": len(ds_pool),
                    "run_dir": run_dir.as_posix(),
                },
            )
            try:
                prep = prepare(cfg, no_download=no_download)
                res = run_trial(
                    prep=prep,
                    cfg=cfg,
                    seeds=seeds,
                    config_sha1=sha1,
                    resume_enabled=bool(resume),
                    write_predictions=False,
                    early_stop=False,
                )
            except Exception as exc:
                record_experiment(
                    Path.cwd(),
                    run_dir=run_dir,
                    status=status_for_run_dir(run_dir, has_summary=False),
                    config_sha1=sha1,
                    config_hash_type="file_sha1",
                    config_raw=resolved_raw,
                    error=str(exc),
                    extra_hparams={"supp.sheet": sheet_key, "supp.target_dataset": target_ds},
                )
                logger.exception(
                    "Supplement evaluation dataset run failed.",
                    extra={"sheet": sheet_key, "target_dataset": target_ds, "run_dir": run_dir.as_posix()},
                )
                raise

            record_experiment(
                Path.cwd(),
                run_dir=res.out_root,
                status=status_for_run_dir(res.out_root, has_summary=True),
                config_sha1=sha1,
                config_hash_type="file_sha1",
                config_raw=resolved_raw,
                summary_df=res.summary_df,
                extra_hparams={"supp.sheet": sheet_key, "supp.target_dataset": target_ds},
            )
            row = res.summary_df.iloc[0].to_dict()
            row["dataset"] = str(row["dataset"]).zfill(4)
            row["target_idx"] = i
            results.append(row)
            logger.info(
                "[%s] %d/%d done: %s mae=%.4f r2=%.4f",
                sheet_key,
                i,
                len(ds_pool),
                target_ds,
                float(row["our_mae_mean"]),
                float(row["our_r2_mean"]),
                extra={
                    "sheet": sheet_key,
                    "target_dataset": target_ds,
                    "target_index": i,
                    "target_count": len(ds_pool),
                    "avg_mae": float(row["our_mae_mean"]),
                    "avg_r2": float(row["our_r2_mean"]),
                    "run_dir": res.out_root.as_posix(),
                },
            )

    our_df = pd.DataFrame(results)
    merged = run_df.merge(
        our_df[["dataset", "our_mae_mean", "our_r2_mean", "delta_mae", "delta_r2", "p_mae", "p_r2", "win_both"]],
        on="dataset",
        how="inner",
    )
    for m in model_names:
        mae_col = f"{m}_mae"
        r2_col = f"{m}_r2"
        if mae_col in merged.columns and r2_col in merged.columns:
            merged[f"better_mae_vs_{m}"] = merged["our_mae_mean"] < merged[mae_col]
            merged[f"better_r2_vs_{m}"] = merged["our_r2_mean"] > merged[r2_col]
            merged[f"better_both_vs_{m}"] = merged[f"better_mae_vs_{m}"] & merged[f"better_r2_vs_{m}"]

    out_dir = out_root / sheet_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "comparison.csv"
    merged.to_csv(out_csv, index=False, encoding="utf-8")

    summary_lines = [f"# Supplement {sheet_key} Single-Task Comparison", ""]
    summary_lines.append(f"- Datasets: {len(merged)}")
    summary_lines.append(f"- Our avg MAE: {merged['our_mae_mean'].mean():.4f}")
    summary_lines.append(f"- Our avg R2: {merged['our_r2_mean'].mean():.4f}")
    for m in model_names:
        bcol = f"better_both_vs_{m}"
        if bcol in merged.columns:
            summary_lines.append(f"- better both vs {m}: {int(merged[bcol].sum())}/{len(merged)}")
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info(
        "Supplement sheet %s completed.",
        sheet_key,
        extra={
            "sheet": sheet_key,
            "comparison_csv": out_csv.as_posix(),
            "summary_md": (out_dir / "summary.md").as_posix(),
            "dataset_count": len(merged),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single-task performance vs Supplementary Table S4/S5")
    parser.add_argument("--supp-xlsx", default="Supp Tables(1).xlsx")
    parser.add_argument("--base-config", default="configs/rplc_14x14_cpvec_hyper_mdlmol_v1.yaml")
    parser.add_argument("--sheet", default="all", help="S4, S5, or all")
    parser.add_argument("--seeds", default="0:4")
    parser.add_argument("--out-root", default="outputs_supp_eval_v1")
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--datasets", default="", help="Optional dataset list filter, e.g. 0052,0180")
    parser.add_argument("--resume", action="store_true", help="Resume from existing per-seed outputs if present")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, or WARNING")
    args = parser.parse_args()

    resolved = resolve_config(args.base_config)
    base_raw = copy.deepcopy(resolved.raw)

    seeds = parse_seed_range(args.seeds)
    out_root = Path(args.out_root)
    xlsx_path = Path(args.supp_xlsx)
    max_ds = int(args.max_datasets) if int(args.max_datasets) > 0 else None
    ds_filter = {str(x).zfill(4) for x in parse_list_expr(args.datasets)} if args.datasets else None
    out_root.mkdir(parents=True, exist_ok=True)
    configure_logging(level=args.log_level, json_log_path=default_run_log_path(out_root, filename="supp_eval.jsonl"))

    sheet_opt = args.sheet.strip().upper()
    if sheet_opt == "ALL":
        sheets = ["Table S4", "Table S5"]
    elif sheet_opt in ("S4", "TABLE S4"):
        sheets = ["Table S4"]
    elif sheet_opt in ("S5", "TABLE S5"):
        sheets = ["Table S5"]
    else:
        sheets = [args.sheet]

    logger.info(
        "Starting supplementary evaluation.",
        extra={
            "supp_xlsx": xlsx_path.as_posix(),
            "base_config": args.base_config,
            "out_root": out_root.as_posix(),
            "sheet_count": len(sheets),
            "seed_count": len(seeds),
            "resume": bool(args.resume),
            "no_download": bool(args.no_download),
        },
    )
    for sh in sheets:
        run_sheet(
            base_raw=base_raw,
            xlsx_path=xlsx_path,
            sheet_name=sh,
            seeds=seeds,
            out_root=out_root,
            no_download=bool(args.no_download),
            resume=bool(args.resume),
            max_datasets=max_ds,
            dataset_filter=ds_filter,
        )


if __name__ == "__main__":
    main()

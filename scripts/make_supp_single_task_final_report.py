from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs_supp_eval_final"
S4_SRC = ROOT / "outputs_supp_eval_s4_combo_v5_uni_5seed" / "S4" / "comparison.csv"
S5_SRC = ROOT / "outputs_supp_eval_s5_combo_v3_final" / "S5" / "comparison.csv"


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"dataset": str}, encoding="utf-8")
    df["dataset"] = df["dataset"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
    return df.sort_values("dataset").reset_index(drop=True)


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _avg_row(sheet: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "sheet": sheet,
        "dataset_count": int(len(df)),
        "our_avg_mae": float(df["our_mae_mean"].mean()),
        "our_avg_r2": float(df["our_r2_mean"].mean()),
        "uni_avg_mae": float(df["Uni_RT_mae"].mean()),
        "uni_avg_r2": float(df["Uni_RT_r2"].mean()),
        "better_both_vs_uni": int(df["better_both_vs_Uni_RT"].sum()),
    }
    if "MDL_TL_mae" in df.columns and "MDL_TL_r2" in df.columns:
        row["mdl_avg_mae"] = float(df["MDL_TL_mae"].mean())
        row["mdl_avg_r2"] = float(df["MDL_TL_r2"].mean())
        if "better_both_vs_MDL_TL" in df.columns:
            row["better_both_vs_mdl"] = int(df["better_both_vs_MDL_TL"].sum())
    return row


def _plot_sheet(sheet: str, df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    x = range(len(df))
    labels = df["dataset"].tolist()

    mae_ax = axes[0]
    mae_ax.plot(x, df["our_mae_mean"], marker="o", linewidth=2, label="Ours", color="#1f77b4")
    mae_ax.plot(x, df["Uni_RT_mae"], marker="s", linewidth=1.8, label="Uni-RT", color="#d62728")
    if "MDL_TL_mae" in df.columns:
        mae_ax.plot(x, df["MDL_TL_mae"], marker="^", linewidth=1.8, label="MDL-TL", color="#2ca02c")
    mae_ax.set_title(f"{sheet} MAE")
    mae_ax.set_xticks(list(x))
    mae_ax.set_xticklabels(labels, rotation=45, ha="right")
    mae_ax.set_ylabel("MAE (sec)")
    mae_ax.grid(alpha=0.25)
    mae_ax.legend()

    r2_ax = axes[1]
    r2_ax.plot(x, df["our_r2_mean"], marker="o", linewidth=2, label="Ours", color="#1f77b4")
    r2_ax.plot(x, df["Uni_RT_r2"], marker="s", linewidth=1.8, label="Uni-RT", color="#d62728")
    if "MDL_TL_r2" in df.columns:
        r2_ax.plot(x, df["MDL_TL_r2"], marker="^", linewidth=1.8, label="MDL-TL", color="#2ca02c")
    r2_ax.set_title(f"{sheet} R2")
    r2_ax.set_xticks(list(x))
    r2_ax.set_xticklabels(labels, rotation=45, ha="right")
    r2_ax.set_ylabel("R2")
    r2_ax.set_ylim(min(0.0, float(df[["our_r2_mean", "Uni_RT_r2"]].min().min()) - 0.05), 1.02)
    r2_ax.grid(alpha=0.25)
    r2_ax.legend()

    fig.suptitle(f"{sheet} Final Single-Task Comparison", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_averages(avg_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    x = range(len(avg_df))
    labels = avg_df["sheet"].tolist()

    mae_ax = axes[0]
    mae_ax.bar([i - 0.25 for i in x], avg_df["our_avg_mae"], width=0.25, label="Ours", color="#1f77b4")
    mae_ax.bar([i for i in x], avg_df["uni_avg_mae"], width=0.25, label="Uni-RT", color="#d62728")
    if "mdl_avg_mae" in avg_df.columns:
        mae_ax.bar([i + 0.25 for i in x], avg_df["mdl_avg_mae"], width=0.25, label="MDL-TL", color="#2ca02c")
    mae_ax.set_title("Average MAE")
    mae_ax.set_xticks(list(x))
    mae_ax.set_xticklabels(labels)
    mae_ax.set_ylabel("MAE (sec)")
    mae_ax.grid(axis="y", alpha=0.25)
    mae_ax.legend()

    r2_ax = axes[1]
    r2_ax.bar([i - 0.25 for i in x], avg_df["our_avg_r2"], width=0.25, label="Ours", color="#1f77b4")
    r2_ax.bar([i for i in x], avg_df["uni_avg_r2"], width=0.25, label="Uni-RT", color="#d62728")
    if "mdl_avg_r2" in avg_df.columns:
        r2_ax.bar([i + 0.25 for i in x], avg_df["mdl_avg_r2"], width=0.25, label="MDL-TL", color="#2ca02c")
    r2_ax.set_title("Average R2")
    r2_ax.set_xticks(list(x))
    r2_ax.set_xticklabels(labels)
    r2_ax.set_ylim(0.0, 1.02)
    r2_ax.grid(axis="y", alpha=0.25)
    r2_ax.legend()

    fig.suptitle("Final Supplement Single-Task Averages", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s4 = _load(S4_SRC)
    s5 = _load(S5_SRC)

    s4.to_csv(OUT_DIR / "s4_final_comparison.csv", index=False, encoding="utf-8")
    s5.to_csv(OUT_DIR / "s5_final_comparison.csv", index=False, encoding="utf-8")

    avg_df = pd.DataFrame([_avg_row("S4", s4), _avg_row("S5", s5)])
    avg_df.to_csv(OUT_DIR / "avg_summary.csv", index=False, encoding="utf-8")

    _plot_sheet("S4", s4, OUT_DIR / "s4_final_vs_baselines.png")
    _plot_sheet("S5", s5, OUT_DIR / "s5_final_vs_baselines.png")
    _plot_averages(avg_df, OUT_DIR / "avg_summary.png")

    lines = [
        "# Final Supplement Single-Task Summary",
        "",
        "## Averages",
    ]
    for row in avg_df.to_dict(orient="records"):
        lines.append(
            f"- {row['sheet']}: our MAE {row['our_avg_mae']:.4f} vs Uni-RT {row['uni_avg_mae']:.4f}; "
            f"our R2 {row['our_avg_r2']:.4f} vs Uni-RT {row['uni_avg_r2']:.4f}; "
            f"better both vs Uni-RT {int(row['better_both_vs_uni'])}/{int(row['dataset_count'])}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            f"- s4 table: `{_rel(OUT_DIR / 's4_final_comparison.csv')}`",
            f"- s5 table: `{_rel(OUT_DIR / 's5_final_comparison.csv')}`",
            f"- avg table: `{_rel(OUT_DIR / 'avg_summary.csv')}`",
            f"- s4 plot: `{_rel(OUT_DIR / 's4_final_vs_baselines.png')}`",
            f"- s5 plot: `{_rel(OUT_DIR / 's5_final_vs_baselines.png')}`",
            f"- avg plot: `{_rel(OUT_DIR / 'avg_summary.png')}`",
        ]
    )
    (OUT_DIR / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

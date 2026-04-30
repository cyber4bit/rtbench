# RepoRT 14x14 Transfer Benchmark

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[test]
```

Cross-platform standard test entrypoint:

```powershell
python -m pytest -q -m "not smoke"
```

`Makefile` is provided for Linux/macOS convenience only. On Windows, use the `python -m pytest ...` commands shown in this README and `docs/dev-guide.md`.

## Full Run

```powershell
python -m rtbench.run --config configs/rplc_14x14.yaml --seeds 0:29
```

Resume is enabled by default. Each run writes `config.sha1` and `config.resolved.yaml` under its output directory. Resume is blocked automatically when the resolved config hash changes.

## Uni-RT-Aligned Evaluation

`configs/vs_unirt.yaml` follows the Uni-RT protocol for the 14 RPLC datasets from Tables S4/Table 1: seeds `0,2,4,6,8,10,12,14,16,18`, random `80/10/10` train/validation/test splits, and the Uni-RT/SOTA baselines in `data/baseline/unirt_sota_28.csv`. `configs/vs_unirt_hilic.yaml` provides the analogous 14-dataset HILIC setup.

```powershell
python -m rtbench.run --config configs/vs_unirt.yaml --no-download
python -m rtbench.report_vs_unirt --run-dir outputs_vs_unirt_rplc --config configs/vs_unirt.yaml
python scripts/plot_vs_unirt.py --run-dir outputs_vs_unirt_rplc --baseline data/baseline/unirt_sota_28.csv --mode RPLC

python -m rtbench.run --config configs/vs_unirt_hilic.yaml --no-download
python -m rtbench.report_vs_unirt --run-dir outputs_vs_unirt_hilic --config configs/vs_unirt_hilic.yaml
python scripts/plot_vs_unirt.py --run-dir outputs_vs_unirt_hilic --baseline data/baseline/unirt_sota_28.csv --mode HILIC
```

The run writes the standard `report.md` plus `report_vs_unirt.md`, Uni-RT summary CSVs under `metrics/`, and figures under `figures/`.

## Config Inheritance And Overrides

Child configs inherit shared bases and keep only the delta:

```yaml
_base:
  - _bases/_base_rplc_14x14.yaml
  - _bases/_base_transfer.yaml
models:
  FUSION_TOP_K: 8
outputs:
  root: outputs_rplc_14x14_topk8
```

Shared bases under `configs/_bases/`:
- `_base_rplc_14x14.yaml`: RepoRT dataset scope, baseline paths, paper metrics, and stats defaults.
- `_base_models_tree.yaml`: standard XGBoost and LightGBM defaults.
- `_base_transfer.yaml`: transfer weighting and target-transform defaults.
- `_base_cpvec.yaml`: CP vectorization defaults for CPEncoder runs.

CLI overrides apply after inheritance:

```powershell
python -m rtbench.run `
  --config configs/rplc_14x14.yaml `
  --override models.FUSION_TOP_K=8 transfer_weights.adaptive_source=true `
  --log-level INFO
```

## CPEncoder Run

```powershell
python -m rtbench.run --config configs/rplc_14x14_cpvec_hyper_best_v2.yaml --seeds 0:29
```

This trains or loads CP vectors from RepoRT metadata and gradient files, then feeds them into the downstream models. Learned artifacts are cached under `data/repoRT/cpvec_cache/`.

## Merge Workflow

```powershell
python -m rtbench.run --config configs/rplc_14x14_cpvec_hyper_mdlmol_v1.yaml --seeds 0:29 --no-download
python -m rtbench.run --config configs/rplc_14x14_cpvec_hyper_mdlmol_0276_n10.yaml --seeds 0:29 --no-download
python -m rtbench.merge_runs `
  --base outputs_rplc_14x14_cpvec_hyper_mdlmol_v1 `
  --override 0276=outputs_tmp_search0276_n10_lam0_ad0 `
  --out outputs_rplc_14x14_cpvec_hyper_mdlmol_plus0276_v1 `
  --baseline-csv data/baseline/paper_table2_3_external_rplc.csv `
  --paper-avg-mae 101.49 --paper-avg-r2 0.84 --required-win-both 8 --fdr-q 0.05
```

Expected merged result in `outputs_rplc_14x14_cpvec_hyper_mdlmol_plus0276_v1/metrics/summary_vs_paper.csv`:
- Avg MAE: `95.6897` (< `101.49`)
- Avg R2: `0.8701` (> `0.84`)
- `win_both`: `8/14`

## Supplementary And Sweep Runs

```powershell
python -m rtbench.experimental.supp_eval --config configs/supp_eval_single_task_v14.yaml --xlsx "Supp Tables(1).xlsx" --sheet S4
python -m rtbench.experimental.sweep --config configs/rplc_14x14_hyper_best.yaml --output-dir outputs_sweep_hyper
```

`rtbench.experimental.sweep` now defaults to `cfg.outputs.root` when `--output-dir` is omitted. `rtbench.experimental.supp_eval` resolves supplementary baselines from `cfg.data.baseline_dir`.

## Outputs

- `outputs.../predictions/{dataset_id}/seed_{k}.csv`
- `outputs.../metrics/per_seed.csv`
- `outputs.../metrics/summary_vs_paper.csv`
- `outputs.../report.md`
- `outputs.../logs/run.jsonl`
- `experiments/registry.csv`

Each runtime entrypoint accepts `--log-level DEBUG|INFO|WARNING`. Structured JSON logs are written under the run output root. For sweep runs, `sweep.jsonl` lives under the chosen sweep output directory and each trial keeps its own `logs/run.jsonl`.

## Experiment Registry

```powershell
python -m rtbench.experiments migrate
python -m rtbench.experiments query --metric avg_mae --sort asc --top 10
python -m rtbench.experiments compare outputs_run_a outputs_run_b
python -m rtbench.experiments gc --status tmp --dry-run
python -m rtbench.experiments cleanup-tmp
```

If `RTBENCH_REGISTRY_PATH` is set, registry commands use that path by default. `--registry` still overrides it explicitly.

Archived duplicate configs are tracked in `experiments/archived_configs.csv`. Historical runs that used a deleted duplicate still resolve through the registry with `archived=true` while preserving the original config path.

## Tests

Linux/macOS convenience commands:

```powershell
make test-unit
make test-smoke
make test-all
make bench
```

Cross-platform equivalents:

```powershell
python -m pytest -q -m "not smoke"
python -m pytest -q -m smoke tests/test_e2e_smoke.py
python -m pytest -q
python -m benchmarks.run_benchmarks --output benchmarks/baseline.json
```

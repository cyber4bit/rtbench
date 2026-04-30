# Developer Guide

## Local Setup

Preferred editable install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[test]
```

Run tests:

```powershell
python -m pytest -q -m "not smoke"
python -m pytest -q -m smoke tests/test_e2e_smoke.py
python -m pytest -q
```

If your shell has `make`, the same commands are exposed as `make test-unit`, `make test-smoke`, and `make test-all`. `Makefile` is Linux/macOS only. On Windows, use the `python -m pytest ...` commands directly.

## Add A New Model Type

CatBoost is a representative example because it follows the tree-style candidate path.

1. Implement the trainer in `rtbench/models/`, for example `rtbench/models/catboost.py`, and make it return `CandidateOutput` objects with validation and test predictions in the same shape used by the existing tree candidates.
2. Register the new builder in `rtbench/models/candidates/__init__.py` so `build_candidates()` remains the single orchestration entrypoint.
3. If the model shares the same target-transform logic as the tree models, mirror the conventions already used in `rtbench.models.trees`. Keep forward transform, inverse transform, and clipping behavior aligned.
4. Add config knobs in a shared or child YAML. Prefer extending `configs/_bases/_base_models_tree.yaml` or a small child config instead of cloning an entire experiment YAML.
5. Wire the config into an experiment YAML and set an explicit `outputs.root` so the run is isolated.
6. Add tests before running the full benchmark.

Minimal CatBoost checklist:

```text
rtbench/models/catboost.py
rtbench/models/candidates/__init__.py
configs/_bases/_base_models_tree.yaml or a child config
tests/test_candidates_*.py or tests/test_ensemble.py
```

Suggested validation flow:

```powershell
python -m pytest -q tests/test_edge_cases.py tests/test_runner.py
python -m rtbench.run --config configs/rplc_14x14_smoke.yaml --override outputs.root=outputs_catboost_smoke
```

## Compare Two Experiments

Use the registry-backed compare command when you need dataset-level diffs between two completed runs.

```powershell
python -m rtbench.experiments compare `
  outputs_rplc_14x14_hyper_best_v1 `
  outputs_rplc_14x14_cpvec_hyper_best_v2
```

The command prints JSON with:
- `run_a` and `run_b`: run-level summary metrics.
- `summary`: overlap counts plus how often run B beats run A on MAE, R2, or both.
- `datasets`: per-dataset MAE and R2 deltas.

Typical follow-up:

```powershell
python -m rtbench.experiments query --metric avg_mae --sort asc --top 5
```

If `RTBENCH_REGISTRY_PATH` is set, both commands use that registry by default. Pass `--registry` to override it.

## Resume After Interruption

Resume safety is enforced by `config.sha1` and the resolved config snapshot saved inside each run directory.

Normal restart:

```powershell
python -m rtbench.run --config configs/rplc_14x14.yaml --seeds 0:29
```

What happens:
- The runner compares the current resolved config hash to `outputs.../config.sha1`.
- If the hash matches, the run can resume.
- If the hash differs, resume is blocked automatically to prevent mixing outputs from different configs.

Manual inspection steps:

```powershell
Get-Content outputs_rplc_14x14_hyper_best_v1\\config.sha1
Get-Content outputs_rplc_14x14_hyper_best_v1\\config.resolved.yaml
python -m rtbench.experiments query --metric avg_mae --sort asc --top 5
```

Common errors and fixes:
- `config.sha1 mismatch`: rerun into a new `outputs.root`, or intentionally remove the old output directory only if you no longer need it.
- Missing `config.resolved.yaml`: the run is incomplete or predates the new snapshot behavior; rebuild the registry and inspect the archived config under `experiments/configs/`.
- Registry points to the wrong config: run `python -m rtbench.experiments migrate` to refresh `experiments/registry.csv`.

## Add A New Dataset

1. Put the RepoRT study files under `data/repoRT/processed_data/<dataset_id>/`.
2. Ensure metadata, gradient, RT, descriptor, and fingerprint files exist.
3. Add the dataset to `datasets.pretrain` or `datasets.external`.
4. Add its baseline row to the configured baseline CSV if the run compares against paper metrics.
5. If CP vectors are enabled, keep metadata and gradient files present because CP vectorization depends on them.

Useful validation path:

```powershell
python -m rtbench.run --config configs/rplc_14x14_smoke.yaml --override outputs.root=outputs_dataset_smoke
```

## Run Hyperparameter Search

For structured sweeps, use the in-process sweep runner:

```powershell
python -m rtbench.experimental.sweep `
  --config configs/rplc_14x14_hyper_best.yaml `
  --seeds 0:9 `
  --output-dir outputs_sweep_hyper `
  --no-download
```

Useful flags:
- `--include-hybrid`: also sweep hybrid fusion settings when the base config includes tree blocks.
- `--write-predictions`: persist per-seed prediction CSVs for every trial.

For one-off ablations, use normal runs plus CLI overrides:

```powershell
python -m rtbench.run `
  --config configs/rplc_14x14_hyper_best.yaml `
  --override models.FUSION_TOP_K=3 transfer_weights.target_transform=logk `
  --seeds 0:9 `
  --no-download
```

## Notes On Config Hygiene

- Prefer stable config names for long-lived experiments.
- Archive temporary search or tuning configs through `experiments/archived_configs.csv` before deleting them.
- Supplementary `supp_s4_*` policy YAML files are plain policy inputs, not benchmark configs, so they intentionally do not use `_base`.

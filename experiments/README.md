# Experiment Registry

`experiments/registry.csv` is the canonical run registry for this repo.

One row represents one runnable output directory:
- a directory that contains `config.sha1`, or
- a directory that contains `metrics/summary_vs_paper.csv`.

This lets the registry capture both top-level benchmark runs and nested single-dataset runs such as `outputs_supp_eval_*/S4/0180`.

## Columns

- `experiment_name`: relative run path, used as the human-readable experiment id.
- `run_dir`: relative path to the actual run root.
- `output_root`: top-level `outputs*` directory that owns the run.
- `run_date`: best-effort completion timestamp from `summary_vs_paper.csv`, else `config.sha1`.
- `status`: `success`, `failed`, or `tmp`.
- `cleanable`: `true` when the top-level output root starts with `outputs_tmp`.
- `archived`: `true` when the original config file was intentionally removed and preserved only as an alias.
- `archived_note`: reason for the archive, usually pointing to the canonical replacement config.
- `config_sha1`: hash found in the run directory.
- `config_hash_type`: `file_sha1` for raw YAML file hashes, `normalized_sha1` for canonicalized config dict hashes.
- `config_path`: best-effort source config path when it can be matched.
- `config_source`: how `config_path` was resolved (`runtime`, catalog match, or heuristic guess).
- `effective_config_path`: canonical snapshot of the fully resolved config under `experiments/configs/`.
- `effective_config_sha1`: canonical SHA1 of the fully resolved config payload.
- `summary_path`: relative path to `summary_vs_paper.csv`.
- `avg_mae`, `avg_r2`: mean of `our_mae_mean` and `our_r2_mean` across datasets.
- `win_both`: count of datasets with `win_both=true`.
- `dataset_count`: datasets included in `summary_vs_paper.csv`.
- `seed_count`: unique seeds found in `per_seed.csv`.
- `key_hparams`: compact JSON of salient hyperparameters and run-scope context.
- `error`: exception text for failed runtime registrations.

## Commands

Rebuild the registry from historical outputs:

```powershell
python -m rtbench.experiments migrate
```

List temporary output roots that are safe to clean:

```powershell
python -m rtbench.experiments cleanup-tmp
```

Query the best runs by a registry metric:

```powershell
python -m rtbench.experiments query --metric avg_mae --sort asc --top 10
```

Compare two runs dataset-by-dataset:

```powershell
python -m rtbench.experiments compare outputs_run_a outputs_run_b
```

Preview registry-backed temporary roots that are safe to garbage-collect:

```powershell
python -m rtbench.experiments gc --status tmp --dry-run
```

Delete those temporary roots and rebuild the registry:

```powershell
python -m rtbench.experiments gc --status tmp --delete
```

Delete all top-level `outputs_tmp*` directories:

```powershell
python -m rtbench.experiments cleanup-tmp --delete
```

The migration command also refreshes `experiments/cleanup_candidates.txt`.

Archived config aliases are stored in `experiments/archived_configs.csv`. During migration, those aliases are matched by historical `config.sha1`, so deleted duplicate YAMLs can still be reconstructed in the registry without restoring the file into `configs/`.

New runs also write a local resolved config snapshot at `outputs.../config.resolved.yaml`.
Registry utilities use Python `logging` too and write `experiments/logs/experiments.jsonl`.

# Pre-Registered Profile Schema (Locked)

Locked at debate Round 5-6. Output of `v4/scripts/profile_failures.py` MUST
match this schema exactly. Adding, renaming, or repurposing columns after L2
finishes is forbidden.

## Output Path

`v4/reports/preregistered_profile.csv`

## Columns

| # | Column | Type | Source | Notes |
| ---: | --- | --- | --- | --- |
| 1 | `dataset` | string, 4-digit zero-padded | dataset id from `v3_vs_uni_rt.csv` | e.g. `0179` |
| 2 | `sheet` | string `S4` or `S5` | filename | |
| 3 | `split_id` | string | concat of (sheet, dataset, seed) | identifies the source split |
| 4 | `n_train` | int | RepoRT processed_data + 81/9/10 split with seed | training sample count |
| 5 | `n_val` | int | as above | |
| 6 | `n_test` | int | as above | |
| 7 | `RT_min` | float (seconds) | training labels | |
| 8 | `RT_max` | float (seconds) | training labels | |
| 9 | `RT_IQR` | float (seconds) | training labels | inter-quartile range |
| 10 | `dup_rate` | float ∈ [0,1] | training SMILES | duplicate fraction; standardized SMILES |
| 11 | `source_CP_distance` | float | CP vector cache | min cosine distance to source dataset CP vectors |
| 12 | `residual_MAE` | float | v3 frozen test predictions | per-dataset, mean across tuning seeds |
| 13 | `residual_R2` | float | v3 frozen test predictions | per-dataset, mean across tuning seeds |
| 14 | `residual_skew` | float | v3 frozen test predictions | per-sample residuals' skewness |
| 15 | `film_grad_ok` | bool or `n/a` | from `cp_audit.json` if available | `true` if CP path has nonzero gradient; `n/a` if no FiLM path exists or audit has not run |
| 16 | `cp_sensitivity_p` | float in [0,1] or blank | from `cp_audit.json` if available | permutation test p-value; read-only, never used by L4 |
| 17 | `failure_tag` | enum | profile script | one of: `win`, `near_miss`, `near_miss_r2`, `near_miss_mae`, `hard_loss`, `catastrophic` |
| 18 | `loss_weight_rule` | float or blank | filled by L4 artifact, not by L2 | left blank in L2 output; rules are written to `loss_weight_rules.json` |

## Derivation Rules

### `failure_tag`

Defined by the pre-registered mapping below and written by
`profile_failures.py`. It is based on v3 frozen metrics on the tuning band, not
on L1 focal output.

```
beat_both == true                       -> win
delta_mae > -2.0 and delta_r2 > -0.01   -> near_miss
delta_mae > -2.0 and delta_r2 <= -0.01  -> near_miss_r2
delta_mae <= -2.0 and delta_r2 > -0.01  -> near_miss_mae
delta_mae < -25.0 or delta_r2 < -0.15   -> catastrophic
otherwise                               -> hard_loss
```

### `loss_weight_rule` (Step 4 only)
Deterministic function of `failure_tag`. Default form:

```
weight = 1.0
if failure_tag in {hard_loss, catastrophic, near_miss_r2, near_miss_mae}:
    weight = 3.0
else:
    weight = 1.0
```

The mapping is locked. If a future debate proposes a different mapping, it must
be added as a new derivation version rather than mutating the v4 default.

## Forbidden Columns

The following must not be added to this schema:

- Any seed-0..9 metric (forbidden tuning leakage).
- Any focal-3x outcome from L1 (the L4 rule must be blind to L1).
- Per-dataset learning-rate suggestions or architecture suggestions.

## Reproducibility

The script writes a sidecar `preregistered_profile.meta.json` containing:

- `rtbench_commit`: SHA of the rtbench module used.
- `tuning_seeds`: `[23, 24, 25]`.
- `cp_cache_path`: where CP vectors were loaded from.
- `v3_outputs_root`: which v3 worktree was scanned.

A subsequent rerun on the same inputs MUST produce a byte-identical CSV (modulo line endings).

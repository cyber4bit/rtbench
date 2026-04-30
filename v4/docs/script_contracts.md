# v4 Script Contracts

These contracts are intentionally concrete. An implementation agent should be
able to write the missing scripts from this document without asking for
strategy clarification.

## Existing Scripts

### `v4/scripts/failure_matrix.py`

Status: implemented.

Required behavior:

- Read S4/S5 v3 `v3_vs_uni_rt.csv`.
- Write `v4/reports/failure_matrix.csv`.
- Write `v4/reports/failure_matrix.md`.
- Exit nonzero if no v3 rows are found.

### `v4/scripts/verify_cp_injection.py`

Status: implemented.

Required behavior:

- Scan target Python files for known v5 anti-patterns.
- Print PASS/FAIL.
- Exit nonzero on any known anti-pattern.

## Missing Scripts To Implement

### `v4/scripts/profile_failures.py`

Purpose: build the locked L2 profile from frozen v3 predictions on seeds
`23..25`. It must not read L1 focal outputs and must not inspect seeds `0..9`.

CLI:

```powershell
python v4/scripts/profile_failures.py `
  --repo-root . `
  --v3-root .claude/worktrees/romantic-wilson-87e23a `
  --tuning-seeds 23,24,25 `
  --out v4/reports/preregistered_profile.csv
```

Reads:

- `<v3-root>/outputs_v3_s4/config.resolved.yaml`
- `<v3-root>/outputs_v3_s4/v3_vs_uni_rt.csv`
- `<v3-root>/outputs_v3_s4/metrics/per_seed.csv`
- `<v3-root>/outputs_v3_s5/config.resolved.yaml`
- `<v3-root>/outputs_v3_s5/v3_vs_uni_rt.csv`
- `<v3-root>/outputs_v3_s5/metrics/per_seed.csv`
- repo data through existing `rtbench.prepare` helpers where needed

Writes:

- `v4/reports/preregistered_profile.csv`
- `v4/reports/preregistered_profile.meta.json`
- `v4/reports/_stamps/profile.ok`

Implementation requirements:

- Output columns must exactly match `docs/preregistered_schema.md`.
- Dataset ids must be four-digit strings.
- `n_train`, `n_val`, `n_test`, `RT_min`, `RT_max`, `RT_IQR`, and `dup_rate`
  are computed from the same 81/9/10 split and seed band.
- `source_CP_distance` is the minimum cosine distance between the target CP
  vector and the pretrain dataset CP vectors.
- `residual_MAE`, `residual_R2`, and `residual_skew` are computed from v3
  predictions for tuning seeds only.
- `failure_tag` is assigned by the locked rule in
  `docs/preregistered_schema.md`.
- `loss_weight_rule` is blank in the CSV.
- If CP audit has not run yet, write `film_grad_ok=n/a` and
  `cp_sensitivity_p` blank; `derive_loss_rules.py` must not use these fields.

Tests:

- Schema exactness test.
- Seed leakage test rejects any input path under L6 or any seed `0..9`.
- Determinism test: two runs produce identical CSV bytes on same inputs.

### `v4/scripts/cp_audit.py`

Purpose: read-only audit of CP injection. It never changes rules or configs.

CLI:

```powershell
python v4/scripts/cp_audit.py `
  --repo-root . `
  --target rtbench `
  --out v4/reports/cp_audit.json
```

Reads:

- Python source under `--target`.
- Optional v3 checkpoint or in-process model if available.

Writes:

- `v4/reports/cp_audit.json`
- `v4/reports/_stamps/cp_audit.ok` only when audit passes or no FiLM is present

JSON schema:

```json
{
  "schema_version": 1,
  "target": "rtbench",
  "film_present": false,
  "film_grad_ok": "n/a",
  "cp_sensitivity_p": null,
  "hard_bug": false,
  "findings": []
}
```

Implementation requirements:

- Reuse `verify_cp_injection.scan()` for static anti-pattern detection.
- If no FiLM or adapter path exists, pass with `film_present=false`.
- If FiLM exists, verify nonzero gradient to CP-conditioned gamma/beta and run
  a simple permutation sensitivity test. Record only p-value; do not feed it
  into `derive_loss_rules.py`.
- `hard_bug=true` if static guard fails or FiLM exists but CP gradient is zero.

Tests:

- Static anti-pattern fixture returns `hard_bug=true`.
- No-FiLM fixture returns pass with `film_grad_ok=n/a`.
- JSON keys are stable.

### `v4/scripts/derive_loss_rules.py`

Purpose: deterministic L4 rule derivation from L2 profile only.

CLI:

```powershell
python v4/scripts/derive_loss_rules.py `
  --profile v4/reports/preregistered_profile.csv `
  --out v4/reports/loss_weight_rules.json
```

Reads:

- `v4/reports/preregistered_profile.csv` only.

Writes:

- `v4/reports/loss_weight_rules.json`
- `v4/reports/loss_weight_rules.md`
- `v4/reports/_stamps/rules.ok`

Rules:

```text
if failure_tag in {hard_loss, catastrophic, near_miss_r2, near_miss_mae}:
    weight = 3.0
else:
    weight = 1.0
```

Implementation requirements:

- Refuse to run if profile path is not `v4/reports/preregistered_profile.csv`
  unless `--allow-noncanonical-profile` is explicitly passed.
- Do not read L1 output, CP audit, L5 output, or L6 output.
- Preserve sheet names in JSON: `rules.S4`, `rules.S5`.
- Omit weight-1.0 datasets from `rules`.

Tests:

- File access test with a monkeypatched `open` guard or code review assert.
- Rule fixture test for each `failure_tag`.
- Rejects unknown `failure_tag`.

### `v4/scripts/run_phase.py`

Purpose: execute L1, L5, and L6 by composing locked v3 configs with phase
manifests.

CLI:

```powershell
python v4/scripts/run_phase.py --phase L1_focal_3x
python v4/scripts/run_phase.py --phase L5_validate
python v4/scripts/run_phase.py --phase L6_final
```

Reads:

- `v4/configs/<phase-manifest>.yaml`
- v3 S4/S5 `config.resolved.yaml`
- `v4/reports/loss_weight_rules.json` for L5/L6 only

Writes:

- `v4/reports/_generated_configs/<phase>_S4.yaml`
- `v4/reports/_generated_configs/<phase>_S5.yaml`
- `outputs_v4_<phase>_S4/`
- `outputs_v4_<phase>_S5/`
- `v4/reports/_stamps/<phase>.ok`

Implementation requirements:

- For L1, use fixed failure-set weights from `v4_focal_3x.yaml`.
- For L5/L6, inject only `loss_weight_rules.json` weights.
- Enforce seed bands from `seed_allocation.yaml`.
- Refuse L6 unless `v4/reports/_stamps/gate.ok` exists.
- Generate configs by applying only whitelist keys to v3 resolved configs.
- Invoke `python -m rtbench.run --config <generated-config>` once per sheet.

Tests:

- Generated config diff test against v3.
- L6 refuses without gate stamp.
- L5 refuses if `rules.ok` stamp is missing.

### `v4/scripts/finalize_v4.py`

Purpose: enforce gates and emit the final report.

CLI:

```powershell
python v4/scripts/finalize_v4.py --gate-check
python v4/scripts/finalize_v4.py --emit-report
```

Reads:

- L5 outputs for `--gate-check`.
- L6 outputs for `--emit-report`.
- `v4/reports/cp_audit.json`.
- `v4/configs/config_diff_whitelist.yaml`.
- `experiments/registry.csv`.

Writes:

- `v4/reports/gate_report.md`
- `v4/reports/v4_summary.csv`
- `v4/reports/v4_report.md`
- `v4/reports/_stamps/gate.ok` on gate pass
- `v4/reports/_stamps/report.ok` on final report pass

Implementation requirements:

- Gate thresholds come from `docs/acceptance_gates.md`.
- Compute `beat_both` from `summary_vs_paper.csv`.
- Check exactly expected seed counts for L6.
- Check generated config diffs are within whitelist.
- If hard bug is present, mark outputs diagnostic only and do not create gate
  or report stamps.

Tests:

- Passing fixture creates `gate.ok`.
- One failed gate blocks `gate.ok`.
- L6 missing seed fails.
- Hard bug fails even if metrics pass.

## Optional rtbench Shim

Only add this if current `rtbench` cannot consume
`transfer_weights.per_dataset_multiplier`.

Files:

- `rtbench/bench/weighting.py`
- `rtbench/bench/runner.py`
- tests under `tests/`

Allowed behavior:

- Scale target training sample weights for the current target dataset.
- Default every unspecified dataset to `1.0`.
- Keep source weighting, validation rows, test rows, architecture, split, and
  optimizer unchanged.

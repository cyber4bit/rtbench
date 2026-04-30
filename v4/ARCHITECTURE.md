# RTBench v4 Architecture

This is the canonical v4 architecture. It converts the debate consensus into a
handoff package that another agent can implement without re-litigating the
strategy.

## 1. Goal

Make v4 pass the three gates on both S4 and S5:

| Sheet | Gate |
| --- | --- |
| S4 | avg MAE below Uni-RT, avg R2 above Uni-RT, `beat_both >= 10/14` |
| S5 | avg MAE below Uni-RT, avg R2 above Uni-RT, `beat_both >= 10/14` |

Known v3 state:

| Sheet | v3 status | Blocking failures |
| --- | --- | --- |
| S4 | avg MAE and avg R2 pass, `beat_both = 9/14` | `0179`, `0180`, `0234`, `0261`, `0264` |
| S5 | avg MAE passes, avg R2 fails, `beat_both = 9/14` | `0027`, `0183`, `0184`, `0185`, `0282` |

## 2. Strategy

v4 is data/loss-first. It keeps the v3 unified HyperNet as the rollback
baseline and applies loss weighting only after a pre-registered failure profile.
Architecture work is out of scope unless:

1. the data/loss path fails,
2. CP audit is clean, and
3. a new debate or written decision expands the v4 scope.

No architecture changes are part of the default v4 pipeline.

## 3. State Machine

```text
START
  |
  v
PRE_FLIGHT
  |-- fail --> STOP: environment or known anti-pattern issue
  |
  v
L1_FIXED_3X  ----\
L2_PROFILE   -----+--> L4_RULES --> L5_VALIDATE --> GATE_CHECK
L3_CP_AUDIT ----/                                  |        |
  |                                                |        v
  |                                                |     L6_FINAL
  |                                                |        |
  |                                                v        v
  |                                             STOP     REPORT
  |
  +-- hard bug --> BUG_BRANCH: diagnostic outputs only, patch, rerun L1-L3
```

Rules:

- L1, L2, and L3 may run in parallel.
- L4 consumes only L2 output.
- L5 may not change L4 rules.
- L6 may not start until gate stamp exists.
- Any hard CP bug invalidates promotion and requires rerun on fixed code.

## 4. Components

| Component | Owner path | Status | Contract |
| --- | --- | --- | --- |
| Failure matrix | `v4/scripts/failure_matrix.py` | implemented | Reads v3 `v3_vs_uni_rt.csv`; writes `failure_matrix.csv/md`. |
| Static CP guard | `v4/scripts/verify_cp_injection.py` | implemented | Scans Python source for known v5 anti-patterns. |
| Failure profile | `v4/scripts/profile_failures.py` | to implement | Writes locked L2 profile CSV and meta JSON. |
| CP audit | `v4/scripts/cp_audit.py` | to implement | Writes read-only audit JSON. |
| Rule derivation | `v4/scripts/derive_loss_rules.py` | to implement | Reads L2 profile only; writes loss rules JSON. |
| Phase runner | `v4/scripts/run_phase.py` | to implement | Builds S4/S5 temp configs from v3 resolved configs and phase manifests. |
| Finalizer | `v4/scripts/finalize_v4.py` | to implement | Checks gates, writes reports, enforces promotion/rollback. |
| Loss-weight shim | `rtbench/bench/weighting.py`, `rtbench/bench/runner.py` | maybe needed | Add only if `per_dataset_multiplier` is missing from active runner. |

Detailed script contracts are in `docs/script_contracts.md`.

## 5. Configuration Architecture

The phase YAML files under `v4/configs/` are phase manifests. `run_phase.py`
must not treat them as standalone rtbench configs. Instead it must:

1. Load the locked v3 resolved config for each sheet:
   - S4: `.claude/worktrees/romantic-wilson-87e23a/outputs_v3_s4/config.resolved.yaml`
   - S5: `.claude/worktrees/romantic-wilson-87e23a/outputs_v3_s5/config.resolved.yaml`
2. Apply only the phase manifest overrides:
   - `seeds.default`
   - `transfer_weights.per_dataset_multiplier`
   - `outputs.root`
3. Write an effective temp config to `v4/reports/_generated_configs/<phase>_<sheet>.yaml`.
4. Run `python -m rtbench.run --config <generated-config>`.

The config diff whitelist is `v4/configs/config_diff_whitelist.yaml`.

## 6. Data Contracts

### 6.1 V3 Inputs

Required paths:

```text
.claude/worktrees/romantic-wilson-87e23a/outputs_v3_s4/config.resolved.yaml
.claude/worktrees/romantic-wilson-87e23a/outputs_v3_s4/v3_vs_uni_rt.csv
.claude/worktrees/romantic-wilson-87e23a/outputs_v3_s4/metrics/per_seed.csv
.claude/worktrees/romantic-wilson-87e23a/outputs_v3_s5/config.resolved.yaml
.claude/worktrees/romantic-wilson-87e23a/outputs_v3_s5/v3_vs_uni_rt.csv
.claude/worktrees/romantic-wilson-87e23a/outputs_v3_s5/metrics/per_seed.csv
```

If these are absent, stop. Do not substitute another worktree silently.

### 6.2 L2 Profile

Output:

```text
v4/reports/preregistered_profile.csv
v4/reports/preregistered_profile.meta.json
```

The CSV schema is locked in `docs/preregistered_schema.md`.

### 6.3 L3 CP Audit

Output:

```text
v4/reports/cp_audit.json
```

Required keys:

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

If no FiLM layer exists in v3, `film_present=false`, `film_grad_ok="n/a"`,
and the audit passes unless static guards find a hard bug.

### 6.4 L4 Loss Rules

Output:

```text
v4/reports/loss_weight_rules.json
```

Canonical shape:

```json
{
  "schema_version": 1,
  "source_profile": "v4/reports/preregistered_profile.csv",
  "rules": {
    "S4": {"0179": 3.0},
    "S5": {"0183": 3.0}
  },
  "default_weight": 1.0,
  "derivation": "failure_tag in hard_loss,catastrophic,near_miss_r2,near_miss_mae -> 3.0"
}
```

Non-failure datasets are omitted and implicitly weight `1.0`.

### 6.5 Phase Outputs

For every run phase and sheet:

```text
outputs_v4_<phase>_<sheet>/config.resolved.yaml
outputs_v4_<phase>_<sheet>/config.sha1
outputs_v4_<phase>_<sheet>/metrics/per_seed.csv
outputs_v4_<phase>_<sheet>/metrics/summary_vs_paper.csv
outputs_v4_<phase>_<sheet>/report.md
```

`finalize_v4.py` also writes:

```text
v4/reports/gate_report.md
v4/reports/v4_summary.csv
v4/reports/v4_report.md
```

## 7. Loss Weighting Contract

If the active runner does not support `transfer_weights.per_dataset_multiplier`,
add the smallest possible shim:

```python
def expand_per_dataset_multiplier(dataset_id: str, mapping: dict[str, float], default: float = 1.0) -> float:
    return float(mapping.get(str(dataset_id).zfill(4), default))
```

Use it only to scale target training rows for the current target dataset. Do
not scale validation or test rows. Do not change model architecture, learning
rate, dropout, hidden dimensions, split, or candidate pool.

Implementation location:

- helper: `rtbench/bench/weighting.py`
- call site: target sample weight assembly in `rtbench/bench/runner.py`

Required tests:

- multiplier defaults to 1.0 when dataset is absent.
- string and integer-like dataset ids normalize to four digits.
- only train target rows get scaled.
- S4/S5 generated configs differ from v3 only by whitelisted keys.

## 8. Gate Logic

L5 to L6 promotion requires:

- S4 avg MAE < 25.5482
- S4 avg R2 > 0.9144
- S4 `beat_both >= 10`
- S5 avg MAE < 48.0916
- S5 avg R2 > 0.8305
- S5 `beat_both >= 10`
- static CP guard passes
- CP audit is clean or `film_present=false`
- config diffs stay inside whitelist

L6 publish requires the same gates on seeds `0..9`, plus:

- exactly 10 seeds per dataset,
- `experiments/registry.csv` contains matching rows,
- config hashes are stable.

## 9. Failure Branches

| Failure | Action |
| --- | --- |
| Missing v3 input | Stop and report missing path. |
| L2 schema mismatch | Stop; fix `profile_failures.py`; rerun L2. |
| L3 hard bug | Mark outputs diagnostic only; patch bug; rerun L1-L3. |
| L5 gate fail | Stop; do not run L6; write `gate_report.md`. |
| L6 gate fail | Rename L6 outputs to `__retracted`; republish v3. |
| Seed leakage found | Invalidate run; delete generated v4 reports; rerun from L2. |

## 10. Completion Definition

The v4 implementation is complete when:

1. all scripts in `docs/script_contracts.md` exist,
2. all tests listed in `docs/agent_handoff.md` pass,
3. L1/L2/L3/L4/L5 run without seed leakage,
4. L5 gate pass creates `_stamps/gate.ok`,
5. L6 creates publishable `v4/reports/v4_report.md`,
6. the report names the exact v3 config hashes and v4 config hashes.

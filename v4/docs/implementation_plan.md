# v4 Implementation Plan

This plan is subordinate to `../ARCHITECTURE.md` and the frozen debate
consensus. It deliberately excludes BoxCox, FiLM, adapters, and MDL fallback
from the default v4 path.

## Phase 0: Pre-flight

Goal: prove the handoff package is pointing at the correct baseline.

Commands:

```powershell
python v4\scripts\failure_matrix.py
python v4\scripts\verify_cp_injection.py --target rtbench
```

Exit criteria:

- `v4/reports/failure_matrix.md` lists exactly 10 failures.
- Static CP guard passes against active `rtbench/`.
- v3 S4/S5 config snapshots and `v3_vs_uni_rt.csv` files exist.

## Phase 1: Implement Missing Orchestration

Goal: create the pipeline shell without changing model behavior.

Implement:

- `profile_failures.py`
- `cp_audit.py`
- `derive_loss_rules.py`
- `run_phase.py`
- `finalize_v4.py`

Add tests for schema exactness, config diff whitelist, seed leakage, and gate
logic. Do not start full training until generated configs are inspected.

## Phase 2: Add Minimal Loss Weight Shim If Needed

Goal: support `transfer_weights.per_dataset_multiplier` while keeping v3
otherwise identical.

Allowed code change:

- Add a helper in `rtbench/bench/weighting.py`.
- Use it in `rtbench/bench/runner.py` to multiply target training weights for
  the current target dataset.

Disallowed:

- Architecture changes.
- Source-weighting changes.
- Split changes.
- Optimizer or HyperNet parameter changes.

## Phase 3: Run L1/L2/L3

Goal: collect independent first-round evidence.

Commands:

```powershell
python v4\scripts\profile_failures.py --v3-root .claude\worktrees\romantic-wilson-87e23a --tuning-seeds 23,24,25 --out v4\reports\preregistered_profile.csv
python v4\scripts\cp_audit.py --target rtbench --out v4\reports\cp_audit.json
python v4\scripts\run_phase.py --phase L1_focal_3x
```

Rules:

- L1 cannot influence L4.
- CP audit cannot influence L4.
- If CP audit finds a hard bug, L1 output is diagnostic only.

## Phase 4: Derive and Validate Rules

Goal: validate the profile-derived rule on held-out seeds.

Commands:

```powershell
python v4\scripts\derive_loss_rules.py --profile v4\reports\preregistered_profile.csv --out v4\reports\loss_weight_rules.json
python v4\scripts\run_phase.py --phase L5_validate
python v4\scripts\finalize_v4.py --gate-check
```

Exit criteria:

- L5 passes all S4/S5 gates.
- Config diffs stay inside whitelist.
- `v4/reports/_stamps/gate.ok` exists.

## Phase 5: Final Evaluation

Goal: publish or reject v4 on final seeds.

Commands:

```powershell
python v4\scripts\run_phase.py --phase L6_final
python v4\scripts\finalize_v4.py --emit-report
```

Exit criteria:

- Seeds `0..9` present for all S4 and S5 datasets.
- S4 and S5 pass avg MAE, avg R2, and `beat_both >= 10/14`.
- `v4/reports/v4_report.md` cites v3 and v4 config hashes.

## Stop Conditions

Stop immediately on:

- missing v3 input paths,
- schema mismatch,
- seed leakage,
- config diff outside whitelist,
- CP hard bug,
- L5 gate failure.

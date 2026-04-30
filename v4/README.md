# RTBench v4 Handoff Package

This folder is the frozen v4 architecture and execution contract derived from
`D:\fenxi-codex\辩论\辩论.md`.

Start here if you are another agent taking over:

1. Read `AGENT_BRIEF.md`.
2. Read `ARCHITECTURE.md`.
3. Read `docs/script_contracts.md`.
4. Implement only the files listed in `docs/agent_handoff.md`.
5. Do not use seeds `0..9` until the L5 gate passes.

## What v4 Is

v4 is a data/loss-first iteration over the v3 unified HyperNet. It does not
start with a new model architecture. The only default intervention is
per-dataset loss weighting, selected through a pre-registered profile and
validated on held-out seeds.

The pipeline is:

```text
L1 fixed 3x focal sanity   seeds 20..22
L2 locked failure profile  seeds 23..25
L3 read-only CP audit      no tuning input
L4 deterministic rules     consumes L2 only
L5 independent validation  seeds 26..29
L6 final evaluation        seeds 0..9 only after gate pass
```

## Current Package Contents

| Path | Role |
| --- | --- |
| `AGENT_BRIEF.md` | Operational brief and execution order. |
| `ARCHITECTURE.md` | Full architecture, state machine, data contracts, and failure branches. |
| `docs/debate_consensus.md` | Frozen debate outcome. |
| `docs/preregistered_schema.md` | Locked CSV schema for L2 and L4. |
| `docs/script_contracts.md` | Exact contracts for scripts that still need implementation. |
| `docs/agent_handoff.md` | Copy-ready task prompt for an implementation agent. |
| `configs/*.yaml` | Phase manifests and guardrail configs. These are read by `run_phase.py`; they are not all direct `rtbench-run` configs. |
| `scripts/failure_matrix.py` | Existing helper that reproduces the 10 failure sets from v3 outputs. |
| `scripts/verify_cp_injection.py` | Existing static guard against known v5 anti-patterns. |

## Non-Negotiable Constraints

- v3 unified HyperNet remains the rollback baseline.
- Per-dataset 81/9/10 split only; merged split is forbidden.
- L4 rule derivation may read only `v4/reports/preregistered_profile.csv`.
- CP audit is read-only and can only invalidate or trigger rerun.
- Hard bug in CP path makes all L5/L6 results diagnostic only.
- Final promotion requires S4 and S5 to pass avg MAE, avg R2, and
  `beat_both >= 10/14` on seeds `0..9`.

## Minimal Pre-flight

Run from repository root:

```powershell
python v4\scripts\failure_matrix.py
python v4\scripts\verify_cp_injection.py --target rtbench
```

The remaining scripts are specified in `docs/script_contracts.md` and must be
implemented before the v4 pipeline can execute end to end.

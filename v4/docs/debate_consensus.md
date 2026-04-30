# RTBench v4 — Frozen Debate Consensus

This file records the **finalized** outcome of the OpenCode-vs-Codex debate (transcript at `D:\fenxi-codex\辩论\辩论.md`). The debate ran 6 rounds and converged early — both sides emitted `【可收口】` at Round 5 and Round 6.

This document is **frozen**. Anything not on this page is out of scope for v4. Any deviation requires a new debate round.

## Final Consensus Table

| Element | Consensus |
| --- | --- |
| Baseline | v3 unified HyperNet retained as rollback |
| Architecture changes | Only allowed after data evidence + audit clean |
| Split | Per-dataset 81/9/10; merged split forbidden |
| Seed bands | 20–22 fixed-3x focal, 23–25 pre-registered profile, 26–29 independent validation, 0–9 final eval only |
| CP audit | Read-only; finds hard bugs but never adjusts hyperparams |
| Hard-bug gate | Bugged 20–29 results are diagnostic only; v4 promotion blocked until rerun on patched code |
| Final eval | Seeds 0–9 against three gates: avg MAE < Uni-RT, avg R² > Uni-RT, beat_both ≥ 10/14 |

## Key Concessions

| Side | Conceded | Won |
| --- | --- | --- |
| OpenCode | Coefficient selection moved off seeds 20–22 onto 26–29 (no on-seed tuning). CP audit accepted as part of L1–L3 parallel pack. Bugged-run gating accepted. | Three-line parallel scheduling kept; "data/loss before architecture" kept; rule selection stays decoupled from audit results. |
| Codex | Audit treated as observational only (no parameter mutation, no rule input). | Pre-registration of failure-tag rules; coefficient validation moved to held-out band; hard-bug → mandatory rerun. Final schema column names locked (`film_grad_ok`, `cp_sensitivity_p`, etc.). |

## What Was Explicitly Ruled Out

The following were proposed during debate and rejected — do not reintroduce them:

1. **Search over focal multiplier on seeds 20–22.** Fixed at 3.0× for failure datasets, 1.0× elsewhere.
2. **Adding columns to the profile table after L2 finishes.** Locked schema only.
3. **Letting CP audit findings feed into rule derivation.** Audit triggers reruns, never inputs.
4. **Architecture changes (FiLM, adapters, LoRA, BoxCox) without prior data evidence and audit pass.**
5. **Per-dataset hyperparameter overrides (lr, dropout, embed_dim).** Whitelist disallows them.
6. **Cherry-picking by examining seeds 0–9 outcomes during tuning.**

## Failure Sets to Rescue (verbatim)

- **S4**: 0179, 0180, 0234, 0261, 0264
- **S5**: 0027, 0183, 0184, 0185, 0282

`loss_weight_rule` derived in L4 may map any subset of these to weight 3.0; non-failure datasets remain at 1.0. The weight is never seeded by L1 or audit results — only by the L2 profile, deterministically.

## Decision Causality (must remain intact)

```
seeds 23–25 v3 frozen forward
        │
        ▼
preregistered_profile.csv (locked schema)
        │
        ▼ (deterministic function only)
loss_weight_rules.json
        │
        ▼
seeds 26–29 validation       ← independent of seeds 20–22 and audit
        │
        ▼
gate check
        │
        ▼ (only on PASS)
seeds 0–9 final
```

If this graph is broken (e.g. rules are derived from L1 results, or L5 is rerun after seeing L6), the v4 result is invalid and must be retracted.

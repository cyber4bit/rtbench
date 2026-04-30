# v4 Bottleneck Review

This file records the bottleneck after the final debate consensus. Earlier
architecture-first proposals are superseded by `debate_consensus.md` and
`../ARCHITECTURE.md`.

## Current v3 Status

| Sheet | Our MAE | Uni-RT MAE | Our R2 | Uni-RT R2 | Beat Both |
| --- | ---: | ---: | ---: | ---: | ---: |
| S4 | 15.2877 | 25.5482 | 0.9701 | 0.9144 | 9/14 |
| S5 | 40.7842 | 48.0916 | 0.8047 | 0.8305 | 9/14 |

## Blocking Failures

| Sheet | Dataset | Failure class from current matrix |
| --- | --- | --- |
| S4 | `0179` | R2 near miss |
| S4 | `0180` | MAE and R2 near miss |
| S4 | `0234` | hard loss |
| S4 | `0261` | hard loss |
| S4 | `0264` | R2 hard loss |
| S5 | `0027` | hard loss |
| S5 | `0183` | catastrophic |
| S5 | `0184` | catastrophic |
| S5 | `0185` | catastrophic |
| S5 | `0282` | R2 catastrophic despite MAE margin |

The bottleneck is not average MAE. It is per-dataset robustness and S5 average
R2.

## Accepted Diagnosis

The debate converged on this diagnosis:

- v3 has enough broad expression power because S4 avg R2 is already strong.
- The losing datasets need more targeted training signal or loss attention.
- A larger architecture would make attribution harder before we know whether a
  simple loss intervention works.
- CP audit is still required because a hard CP path bug would make any profile
  involving CP distance uninterpretable.

## Accepted First Intervention

The only default v4 intervention is per-dataset loss weighting:

- L1: fixed 3x on all known failure datasets, seeds `20..22`, sanity only.
- L2: locked profile table, seeds `23..25`.
- L4: deterministic rules from L2 only.
- L5: held-out validation, seeds `26..29`.
- L6: final seeds `0..9` only after L5 gate pass.

## Explicitly Deferred

These are not part of default v4:

- BoxCox or other target transforms.
- CP-FiLM architecture additions.
- adapters or CrossStitch-like modules.
- MDL-subset fallback.
- per-dataset model-family or hyperparameter search.

They can be revisited only after v4's data/loss path is completed or formally
rejected.

# v6 Reseal Validate Gate Report

| Phase | Sheet | Check | Pass | Detail |
| --- | --- | --- | --- | --- |
| V6_reseal_validate | S4 | avg_mae | PASS | 12.5445 < 25.5482 |
| V6_reseal_validate | S4 | avg_r2 | PASS | 0.9768 > 0.9144 |
| V6_reseal_validate | S4 | beat_both | PASS | 10 >= 10 |
| V6_reseal_validate | S4 | seed_count | PASS | exact expected seed set |
| V6_reseal_validate | S4 | config_diff | PASS | within whitelist |
| V6_reseal_validate | S5 | avg_mae | PASS | 47.5388 < 48.0916 |
| V6_reseal_validate | S5 | avg_r2 | FAIL | 0.7014 > 0.8305 |
| V6_reseal_validate | S5 | beat_both | FAIL | 5 >= 10 |
| V6_reseal_validate | S5 | seed_count | PASS | exact expected seed set |
| V6_reseal_validate | S5 | config_diff | PASS | within whitelist |
| V6_reseal_validate | ALL | cp_guard_and_audit | PASS | PASS |

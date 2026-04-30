# v4 Failure Matrix

- Total rows: 28
- Failures: 10

| Sheet | Dataset | Kind | Severity | Delta MAE | Delta R2 |
| --- | --- | --- | --- | ---: | ---: |
| S4 | 0179 | r2_only_loss | near_miss | 1.0717 | -0.0041 |
| S4 | 0180 | mae_and_r2_loss | near_miss | -1.2263 | -0.0093 |
| S4 | 0234 | mae_and_r2_loss | hard_loss | -6.1492 | -0.0128 |
| S4 | 0261 | mae_and_r2_loss | hard_loss | -16.0603 | -0.0214 |
| S4 | 0264 | r2_only_loss | hard_loss | 3.7947 | -0.0289 |
| S5 | 0027 | mae_and_r2_loss | hard_loss | -2.0292 | -0.0305 |
| S5 | 0183 | mae_and_r2_loss | catastrophic | -77.7305 | -0.4785 |
| S5 | 0184 | mae_and_r2_loss | catastrophic | -60.5252 | -0.3136 |
| S5 | 0185 | mae_and_r2_loss | catastrophic | -25.8509 | -0.1641 |
| S5 | 0282 | r2_only_loss | catastrophic | 9.7917 | -0.3660 |

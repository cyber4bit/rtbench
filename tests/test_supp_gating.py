from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rtbench.experimental.supp_gating import _choose_best_available, _evaluate_policy, _utility


class TestSuppGating(unittest.TestCase):
    def test_utility_prefers_win_both(self):
        row_win = pd.Series(
            {
                "MDL_TL_mae": 10.0,
                "MDL_TL_r2": 0.80,
                "our_mae_mean": 9.0,
                "our_r2_mean": 0.81,
            }
        )
        row_no_win = pd.Series(
            {
                "MDL_TL_mae": 10.0,
                "MDL_TL_r2": 0.80,
                "our_mae_mean": 9.5,
                "our_r2_mean": 0.79,
            }
        )
        u1 = _utility(row_win, baseline_model="MDL_TL", weight_win=4.0, weight_mae=1.0, weight_r2=1.0)
        u2 = _utility(row_no_win, baseline_model="MDL_TL", weight_win=4.0, weight_mae=1.0, weight_r2=1.0)
        self.assertGreater(u1, u2)

    def test_choose_best_available_respects_mask(self):
        classes = np.array([0, 1, 2], dtype=int)
        id_to_run = {0: "v1", 1: "v14", 2: "hn"}
        probs = np.array([0.1, 0.8, 0.1], dtype=float)
        pick = _choose_best_available(
            classes=classes,
            probs=probs,
            available={"v1", "hn"},
            fallback="v1",
            id_to_run=id_to_run,
        )
        self.assertEqual(pick, "v1")

    def test_evaluate_policy_counts(self):
        run_v1 = pd.DataFrame(
            [
                {
                    "dataset": "0001",
                    "our_mae_mean": 9.0,
                    "our_r2_mean": 0.90,
                    "MDL_TL_mae": 10.0,
                    "MDL_TL_r2": 0.80,
                },
                {
                    "dataset": "0002",
                    "our_mae_mean": 11.0,
                    "our_r2_mean": 0.70,
                    "MDL_TL_mae": 10.0,
                    "MDL_TL_r2": 0.80,
                },
            ]
        ).set_index("dataset")
        run_v2 = pd.DataFrame(
            [
                {
                    "dataset": "0001",
                    "our_mae_mean": 9.5,
                    "our_r2_mean": 0.85,
                    "MDL_TL_mae": 10.0,
                    "MDL_TL_r2": 0.80,
                },
                {
                    "dataset": "0002",
                    "our_mae_mean": 8.0,
                    "our_r2_mean": 0.86,
                    "MDL_TL_mae": 10.0,
                    "MDL_TL_r2": 0.80,
                },
            ]
        ).set_index("dataset")
        ev = _evaluate_policy(policy={"0001": "v1", "0002": "v2"}, run_idx={"v1": run_v1, "v2": run_v2})
        self.assertEqual(ev.n, 2)
        self.assertEqual(ev.better_both.get("MDL_TL"), 2)


if __name__ == "__main__":
    unittest.main()

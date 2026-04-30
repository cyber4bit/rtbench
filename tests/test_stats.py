from __future__ import annotations

import unittest

import pandas as pd

from rtbench.stats import bh_fdr, summarize_vs_paper, wilcoxon_greater


class TestStatsBoundaries(unittest.TestCase):
    def test_wilcoxon_greater_returns_one_for_all_zero_differences(self):
        self.assertEqual(wilcoxon_greater([0.0, 0.0, 0.0]), 1.0)

    def test_bh_fdr_handles_empty_and_boundary_values(self):
        self.assertEqual(bh_fdr([]), [])
        self.assertEqual(bh_fdr([0.0, 1.0, 1.0]), [0.0, 1.0, 1.0])

    def test_summarize_vs_paper_uses_strict_fdr_threshold(self):
        per_seed = pd.DataFrame(
            [
                {"dataset": "0001", "seed": 0, "mae": 9.0, "r2": 0.60},
            ]
        )
        baseline = pd.DataFrame(
            [
                {"dataset": "0001", "paper_mae": 10.0, "paper_r2": 0.50},
            ]
        )

        summary = summarize_vs_paper(per_seed_df=per_seed, baseline_df=baseline, fdr_q=0.5)

        self.assertAlmostEqual(float(summary.loc[0, "p_mae"]), 0.5, places=6)
        self.assertAlmostEqual(float(summary.loc[0, "p_r2"]), 0.5, places=6)
        self.assertAlmostEqual(float(summary.loc[0, "p_adj_mae"]), 0.5, places=6)
        self.assertAlmostEqual(float(summary.loc[0, "p_adj_r2"]), 0.5, places=6)
        self.assertFalse(bool(summary.loc[0, "win_both"]))

    def test_summarize_vs_paper_keeps_zero_delta_pvalues_at_one(self):
        per_seed = pd.DataFrame(
            [
                {"dataset": "0001", "seed": 0, "mae": 10.0, "r2": 0.50},
                {"dataset": "0001", "seed": 1, "mae": 10.0, "r2": 0.50},
            ]
        )
        baseline = pd.DataFrame(
            [
                {"dataset": "0001", "paper_mae": 10.0, "paper_r2": 0.50},
            ]
        )

        summary = summarize_vs_paper(per_seed_df=per_seed, baseline_df=baseline, fdr_q=0.05)

        self.assertEqual(float(summary.loc[0, "p_mae"]), 1.0)
        self.assertEqual(float(summary.loc[0, "p_r2"]), 1.0)
        self.assertEqual(float(summary.loc[0, "p_adj_mae"]), 1.0)
        self.assertEqual(float(summary.loc[0, "p_adj_r2"]), 1.0)
        self.assertFalse(bool(summary.loc[0, "win_both"]))


if __name__ == "__main__":
    unittest.main()

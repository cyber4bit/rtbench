from __future__ import annotations

import unittest
from unittest.mock import patch

from benchmarks.run_benchmarks import run_all_benchmarks


class TestBenchmarks(unittest.TestCase):
    @patch("benchmarks.run_benchmarks.run_data_loading_benchmark", return_value={"mean_ms": 3.0})
    @patch("benchmarks.run_benchmarks.run_cpvec_benchmark", return_value={"mean_ms": 2.0})
    @patch("benchmarks.run_benchmarks.run_candidate_builder_benchmark", return_value={"mean_ms": 1.0})
    def test_run_all_benchmarks_collects_all_sections(
        self,
        _candidate_builder_mock,
        _cpvec_mock,
        _data_loading_mock,
    ) -> None:
        payload = run_all_benchmarks()
        self.assertIn("generated_at", payload)
        self.assertIn("platform", payload)
        self.assertIn("python", payload)
        self.assertEqual(payload["benchmarks"]["candidate_builder"]["mean_ms"], 1.0)
        self.assertEqual(payload["benchmarks"]["cpvec"]["mean_ms"], 2.0)
        self.assertEqual(payload["benchmarks"]["data_loading"]["mean_ms"], 3.0)


if __name__ == "__main__":
    unittest.main()

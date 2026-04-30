from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path

from rtbench.logging_utils import attach_json_log, configure_logging, default_run_log_path


class TestLoggingUtils(unittest.TestCase):
    def tearDown(self) -> None:
        configure_logging(level="WARNING", console=False)

    def test_configure_logging_writes_structured_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "outputs_demo"
            log_path = default_run_log_path(run_dir)
            logger = configure_logging(level="INFO", json_log_path=log_path, console=False)

            child = logging.getLogger("rtbench.tests.logging")
            child.info("hello world", extra={"dataset": "0028", "seed": 3})
            for handler in logger.handlers:
                handler.flush()

            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["level"], "INFO")
            self.assertEqual(rows[0]["message"], "hello world")
            self.assertEqual(rows[0]["fields"]["dataset"], "0028")
            self.assertEqual(rows[0]["fields"]["seed"], 3)
            self.assertEqual(rows[0]["logger"], "rtbench.tests.logging")
            configure_logging(level="WARNING", console=False)

    def test_attach_json_log_writes_trial_specific_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sweep_log = default_run_log_path(root / "outputs_sweep", filename="sweep.jsonl")
            trial_log = default_run_log_path(root / "outputs_sweep" / "trial_a")
            logger = configure_logging(level="INFO", json_log_path=sweep_log, console=False)
            child = logging.getLogger("rtbench.tests.logging")

            with attach_json_log(trial_log):
                child.info("inside trial", extra={"trial": "trial_a"})
                for handler in logger.handlers:
                    handler.flush()

            child.info("outside trial", extra={"trial": "summary"})
            for handler in logger.handlers:
                handler.flush()

            sweep_rows = [json.loads(line) for line in sweep_log.read_text(encoding="utf-8").splitlines() if line.strip()]
            trial_rows = [json.loads(line) for line in trial_log.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual([row["message"] for row in sweep_rows], ["inside trial", "outside trial"])
            self.assertEqual([row["message"] for row in trial_rows], ["inside trial"])
            self.assertEqual(trial_rows[0]["fields"]["trial"], "trial_a")
            configure_logging(level="WARNING", console=False)


if __name__ == "__main__":
    unittest.main()

# Linux/macOS only. Windows users should run the documented `python -m pytest ...` commands instead.
.PHONY: test-unit test-smoke test-all bench

test-unit:
	python -m pytest -q -m "not smoke"

test-smoke:
	python -m pytest -q -m smoke tests/test_e2e_smoke.py

test-all:
	python -m pytest -q

bench:
	python -m benchmarks.run_benchmarks --output benchmarks/baseline.json

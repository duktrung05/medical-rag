.PHONY: test test-vibiomir test-unit test-integration test-regression test-repository validate lint format clean

test:
	pytest tests/ -v

test-vibiomir:
	pytest tests/workflows/vibiomir -v

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

test-regression:
	pytest tests/workflows/vimedqa tests/workflows/medquad -v

test-repository:
	pytest tests/repository -v


lint:
	ruff check .

format:
	ruff format .

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"

.PHONY: test validate lint format clean

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"

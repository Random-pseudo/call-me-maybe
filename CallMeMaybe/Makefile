.PHONY: install run debug clean lint

install:
uv sync --all-extras

run:
uv run python -m src

debug:
uv run python -m pdb -m src

clean:
find . -type d -name "pycache" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .mypy_cache
rm -rf .uv
@echo "Clean completed."

lint:
uv run flake8 src
uv run mypy --strict src
.PHONY: mac run test clean

# Apple Silicon path: MLX-accelerated ASR (parakeet-mlx).
# Uses the `mac` extra in pyproject.toml.
mac:
	uv sync --extra mac --extra dev && uv run phoneme-download --profile mac

# Launch the app. Bypass the user's shell $VIRTUAL_ENV (often points at a
# system Python) by invoking the venv's python directly.
run:
	.venv/bin/python -m phoneme

test:
	.venv/bin/python -m pytest

clean:
	rm -rf .venv

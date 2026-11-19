.DEFAULT_GOAL := help
.PHONY: help install install-local gather-results check-uv

help:
	@echo "make install         Install core dependencies (API inference: OpenAI, Azure, Gemini/Vertex AI, AWS Bedrock)"
	@echo "make install-local   Install core + vllm for local inference (vllm is installed on Linux only)"
	@echo "make gather-results  Summarize evaluation/results/ into gathered_results.csv and report.md there"

check-uv:
	@command -v uv >/dev/null 2>&1 || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }

install: check-uv
	uv sync

install-local: check-uv
	uv sync --group local

gather-results: check-uv
	uv run python evaluation/gather_results.py

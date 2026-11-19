.DEFAULT_GOAL := help
.PHONY: help install install-local check-uv

help:
	@echo "make install        Install core dependencies (API inference: OpenAI, Azure, Gemini/Vertex AI, AWS Bedrock)"
	@echo "make install-local  Install core + vllm for local inference (vllm is installed on Linux only)"

check-uv:
	@command -v uv >/dev/null 2>&1 || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }

install: check-uv
	uv sync

install-local: check-uv
	uv sync --group local

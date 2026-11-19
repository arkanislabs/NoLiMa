# pyproject.toml with dependency groups — design

**Status:** approved 2026-11-19 · **Size:** small change · **Branch:** `first-run`

## Intent

Replace `requirements.txt` with a `pyproject.toml` so a machine that only calls hosted
APIs (e.g. a Mac) can install without vllm, while a Linux GPU box can add vllm for local
inference. Installation goes through `make` targets.

## Goals / non-goals

- **Goal:** `make install` gives a working API-inference environment on macOS and Linux.
- **Goal:** `make install-local` adds vllm on Linux.
- **Goal:** reproducible installs via a committed `uv.lock`.
- **Non-goal:** changing evaluation code, or upgrading pinned versions.
- **Anti-goal:** an install that succeeds but crashes on import (see constraints).

## Constraints

- `evaluation/async_api_connector.py` imports every API SDK (openai, google-genai,
  vertexai, langchain-aws, transformers, tiktoken) at module top level, so all of them are
  needed for any run, including vllm runs (vllm is reached through the OpenAI client).
- Pinned versions (tiktoken 0.7.0, transformers 4.46.1, …) have no Python 3.14 wheels.
- vllm requires Linux + NVIDIA GPU.
- `uv` 0.13.1 is installed on the dev machine.

## Approaches

1. **Three groups (core / api / local)** as first requested. Rejected: because of the
   top-level imports, core without api cannot run anything; it would need code changes
   (lazy imports) to make the split meaningful.
2. **Lazy imports + per-provider groups.** Smallest installs, but changes evaluation code,
   which is a non-goal.
3. **API dependencies in core, vllm in a `local` group.** *Chosen* (user's proposal): no
   code changes, matches what the code needs today. Trade-off: every install includes the
   Google Cloud and AWS SDKs, same as `requirements.txt` did.

## Design

- `pyproject.toml`
  - `[project].dependencies` (core): every `requirements.txt` entry except vllm, same
    pins, plus `numpy` and `tqdm` (used directly, previously only transitive).
  - `[dependency-groups] local = ["vllm; sys_platform == 'linux'"]`.
  - `requires-python = ">=3.10,<3.13"`; `.python-version` = `3.12`.
  - `[tool.uv] package = false` (scripts repo, not an installable package).
- `uv.lock` committed.
- `Makefile`: `install` → `uv sync`; `install-local` → `uv sync --group local`.
- Delete `requirements.txt`; update the install section of `README.md`.

## Left to the builder (Claude's calls)

- vllm left unpinned; the resolver picks a version compatible with
  `transformers==4.46.1` (expected ~0.6.x). Newer vllm would require raising that pin.
- Exact Makefile wording/help text and README phrasing.

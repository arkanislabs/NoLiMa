# pyproject.toml Dependency Groups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `requirements.txt` with a uv-managed `pyproject.toml` (core deps + `local` group for vllm) and `make install` / `make install-local` targets.

**Architecture:** Core dependencies live in `[project].dependencies`; vllm lives in a PEP 735 dependency group `local`, restricted to Linux by marker. `uv.lock` is committed. A Makefile wraps `uv sync`.

**Tech Stack:** uv 0.13.1, Python 3.12, GNU/BSD make.

**Spec:** `docs/superpowers/specs/2026-11-19-pyproject-dependency-groups-design.md`

## Global Constraints

- Core pins copied verbatim from `requirements.txt`: `openai==1.53.0`, `tiktoken==0.7.0`, `transformers==4.46.1`, `google-genai==0.3.0`, `google-cloud-aiplatform==1.74.0`, `langchain-aws==0.2.9`, `langchain-core==0.3.26`, `tenacity==9.0.0`, `jsonargparse==4.34.0`, `httpx<0.28`; plus unpinned `numpy`, `tqdm`.
- `local` group: `vllm; sys_platform == 'linux'` (unpinned).
- `requires-python = ">=3.10,<3.13"`; `.python-version` contains `3.12`.
- `[tool.uv] package = false`.
- No changes to evaluation or data code.

## Review Focus

- `uv` not installed → `make install` exits non-zero with a message saying to install uv (link to https://docs.astral.sh/uv/), not a bare "command not found". Test in Task 2.
- Existing `.venv` on Python 3.14 → `make install` replaces it with 3.12, no manual deletion. Test in Task 1.
- `make install-local` on macOS → succeeds and installs no vllm. Test in Task 2.
- `make install` after `make install-local` → vllm is removed (`uv sync` is exact). Call this out in the README. Checked in Task 2.
- Running the eval after install → `evaluation/run_tests.py` and `data/book_haystack.py` import cleanly in the synced env. Test in Task 1.

---

### Task 1: pyproject.toml, Python pin, lockfile

**Files:**
- Create: `pyproject.toml`, `.python-version`, `uv.lock` (generated)

**Interfaces:**
- Produces: a lockable project where `uv sync` = core and `uv sync --group local` = core + vllm (Linux only).

- [ ] **Step 1: Verify the import check fails today**

Run: `cd evaluation && PYTHONPATH=.. ../.venv/bin/python -c "import run_tests, data.book_haystack"`
Expected: FAIL with `ModuleNotFoundError` (the venv is empty).

- [ ] **Step 2: Write `pyproject.toml` and `.python-version`**

`[project]` name `nolima`, version `0.1.0`, the Global Constraints' `requires-python` and core dependencies; `[dependency-groups] local = [...]`; `[tool.uv] package = false`.

- [ ] **Step 3: Lock and check the vllm resolution**

Run: `uv lock && uv export --group local --no-hashes | grep -E '^(vllm|transformers)=='`
Expected: `transformers==4.46.1` and a `vllm==0.6.x ; sys_platform == 'linux'` line. If locking fails because of a vllm/transformers conflict, stop and report. Don't change the pins.

- [ ] **Step 4: Sync and run the import check**

Run: `uv sync && .venv/bin/python --version && cd evaluation && PYTHONPATH=.. ../.venv/bin/python -c "import run_tests, data.book_haystack; print('ok')"`
Expected: `Python 3.12.x`, then `ok` (a transformers "PyTorch not found" warning is fine).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .python-version uv.lock
git commit -m "Add pyproject.toml with core deps and local (vllm) group"
```

### Task 2: Makefile, remove requirements.txt, README

**Files:**
- Create: `Makefile`
- Delete: `requirements.txt`
- Modify: `README.md:74-78` (section "1. Installing Requirements")

**Interfaces:**
- Consumes: Task 1's `pyproject.toml` / `uv.lock`.
- Produces: `make install` → `uv sync`; `make install-local` → `uv sync --group local`; `make help` (default target) lists both.

- [ ] **Step 1: Verify the targets don't exist yet**

Run: `make install`
Expected: FAIL with `No rule to make target` (or `No targets specified`).

- [ ] **Step 2: Write the `Makefile`**

Targets `help` (default), `install`, `install-local`, all `.PHONY`. Both install targets first check `command -v uv` and, if it's missing, exit 1 with "uv is required: https://docs.astral.sh/uv/getting-started/installation/".

- [ ] **Step 3: Verify the targets**

Run: `make install && make install-local && .venv/bin/python -c "import importlib.util as u; print(u.find_spec('vllm'))"`
Expected: both succeed; prints `None` on macOS.

Run: `PATH=/usr/bin:/bin make install; echo "exit=$?"`
Expected: the uv-required message and `exit=2` (make reports a failed recipe as 2).

- [ ] **Step 4: Delete `requirements.txt` and update the README**

Replace `pip install -r requirements.txt` with: install uv, run `make install` (API models), or `make install-local` on a Linux GPU machine (adds vllm); then `source .venv/bin/activate` before running the scripts. Note that `make install` removes vllm if it was installed.

- [ ] **Step 5: Commit**

```bash
git add Makefile README.md requirements.txt
git commit -m "Add Makefile install targets, drop requirements.txt"
```

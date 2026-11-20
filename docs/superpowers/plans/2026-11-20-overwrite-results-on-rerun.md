# Overwrite Results on Re-run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the "skip a test if its result file already exists" logic. Each test instead writes one file with a fixed name and overwrites it on every run. The request cache makes re-runs and resumed runs free for requests already answered.

**Architecture:** `NoLiMa_Tester` names its result file from the test's identity (model, test, haystack, context length) instead of a timestamp, and writes it atomically, replacing any previous version. The test-hash scan (`prevent_duplicate`, `get_hash`, `test_hash`) is deleted. `gather_results.py` stops depending on `test_hash` and warns when a run holds more than one file for the same test, haystack and context length. That can only come from old timestamped files left next to new ones.

**Tech Stack:** Python 3.12, uv, pytest (`make test`).

**Spec:** No separate spec file. This follows on from the request-cache work (`docs/superpowers/plans/2026-11-20-llm-request-cache.md`, already implemented on `first-run`). The user's request (2026-11-20): *"so once we have the cache we can remove any logic that checks if the results files from run_tests are already there to decide what to skip, can we?"* They answered "yes" to this approach: fixed file name per test, overwritten on re-run, so there are no duplicates for `gather_results` to double-count.

## Why the files can't just be skipped-or-kept as they are

- `gather_results.py` loads every result file it finds. With timestamped names and no skip, every re-run or resume would add a second file per test. The report would then double-count requests, "correct / requests" and total cost.
- The old check also hid a bug. `test_hash` ignores the model config, so changing e.g. `max_tokens` and re-running silently skipped every test. With overwrite, the new settings run (new requests, so a cache miss) and replace the old results.
- **Accepted cost:** a re-run re-tokenizes the haystack for every test, which skipping avoided. This takes time but no money.

## Global Constraints

- Result file stem: `"_".join(part for part in (model_name, "book", test_name, haystack_name, f"CL{context_length}") if part)`, where `haystack_name` = the haystack file's base name without its extension. Example: `deepseek_v4.1_flash_1017_openrouter_book_0401_T17_C02_onehop_rand_book_1_CL250`. `eval_name` (stored in the file, and used by placement logs) equals this stem.
- Result files are written with `write_json_atomic(path, outputs, indent=4)` (temp file + `os.replace`), so a crash mid-write leaves the previous version intact.
- Removed entirely: `NoLiMa_Tester(prevent_duplicate=...)`, `NoLiMa_Tester.get_hash`, the `test_hash` result field, the `--prevent_duplicate_tests` option of `run_tests.py`, and the "Results already exist … Skipping evaluation" early return.
- Request cache keys and entries do not change.
- Existing timestamped result files are never deleted or renamed by code. Cleaning them up is the user's call.
- Commit messages follow the repo style (a plain imperative sentence, no `feat:` prefix) and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- An interrupted run resumed into the same folder: tests that already finished are rebuilt from the cache (0 sent), the rest are sent, and there is one file per test. Test: Task 1 `test_failed_request_keeps_successful_responses_in_cache` (now resumes into the same folder).
- Changing the model config (e.g. `max_tokens`) and re-running: new requests are sent and the file is overwritten with the new config, where the old behavior silently skipped. Test: Task 1 `test_changed_model_config_runs_again`.
- A crash while writing a result file: the previous version stays intact and no temp file is left behind. Test: Task 1 `test_write_json_atomic_keeps_the_old_file_when_writing_fails`.
- Old timestamped files next to new fixed-name ones: nothing is deleted, and the report warns that those tests are counted more than once. Test: Task 2 `test_report_warns_about_repeated_result_files`.
- `async_evaluate.py` run with no `test_name`: the name has no empty segment (`m_book_b_CL250`, not `m_book__b_CL250`). Test: Task 1 `test_result_file_name`.

---

### Task 1: Fixed result file names, overwrite on re-run, remove the duplicate check

**Files:**
- Modify: `evaluation/request_cache.py` (add `write_json_atomic`; `RequestCache.put` uses it)
- Modify: `evaluation/async_evaluate.py` (add `result_file_name`; `__init__`, `evaluate`; delete `get_hash`, `prevent_duplicate`; drop imports nothing uses any more, such as `time`, `hashlib`, `deepcopy`)
- Modify: `evaluation/run_tests.py` (remove `--prevent_duplicate_tests` and its pass-through)
- Modify: `README.md` (the "Collect the results" bullet and the "Request cache" subsection's `prevent_duplicate_tests` bullet)
- Modify: `tests/test_request_cache.py`, `tests/test_async_evaluate.py`

**Interfaces:**
- Produces:
  - `request_cache.write_json_atomic(path: str, data: Any, **dump_kwargs) -> None`: `json.dump(data, f, **dump_kwargs)` into `<path>.tmp-<pid>-<uuid4 hex>` in the same directory, then `os.replace`. The temp file is removed if anything raises. `RequestCache.put` calls it with `indent=1, ensure_ascii=False`, as it writes today.
  - `async_evaluate.result_file_name(model_name: str, test_name: str, haystack_path: str, context_length: int) -> str`: the stem from Global Constraints
  - `NoLiMa_Tester.__init__` without `prevent_duplicate`. `self.eval_name = result_file_name(...)`.
  - `NoLiMa_Tester.evaluate() -> list`: always runs and always returns the per-request results. If the file already exists, print `Overwriting {results_path}` before writing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_request_cache.py` (add `import pytest` and `write_json_atomic` to its imports):

```python
def test_write_json_atomic_replaces_the_file(tmp_path):
    path = tmp_path / "r.json"
    write_json_atomic(str(path), {"a": 1})
    write_json_atomic(str(path), {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2} and os.listdir(tmp_path) == ["r.json"]


def test_write_json_atomic_keeps_the_old_file_when_writing_fails(tmp_path):
    path = tmp_path / "r.json"
    write_json_atomic(str(path), {"a": 1})
    with pytest.raises(TypeError):
        write_json_atomic(str(path), {"a": object()})
    assert json.loads(path.read_text()) == {"a": 1} and os.listdir(tmp_path) == ["r.json"]
```

In `tests/test_async_evaluate.py`:
- Import `result_file_name` next to `NoLiMa_Tester`.
- Remove `"test_hash"` from `RESULT_KEYS` and change its comment to "Top-level keys of a result file". In `test_second_run_is_served_from_cache`, drop the trailing `# nothing new feeds test_hash` comment.
- Change `test_failed_request_keeps_successful_responses_in_cache` to resume into the same folder: both calls use `str(tmp_path / "run")`. Keep its existing asserts, then add `assert len(list((tmp_path / "run").glob("*.json"))) == 1`.
- Add:

```python
def test_result_file_name():
    assert result_file_name("m", "t", "/data/rand_book_1.txt", 1000) == "m_book_t_rand_book_1_CL1000"
    assert result_file_name("m", "", "b.txt", 250) == "m_book_b_CL250"


def test_rerun_overwrites_the_result_file(make_tester, sent, tmp_path):
    make_tester(str(tmp_path / "run")).evaluate()
    second = make_tester(str(tmp_path / "run")).evaluate()
    assert len(sent) == 3
    assert [p.name for p in (tmp_path / "run").iterdir()] == ["fake_book_t_book_CL200.json"]
    saved = json.loads((tmp_path / "run" / "fake_book_t_book_CL200.json").read_text())
    assert saved["eval_name"] == "fake_book_t_book_CL200"
    assert [r["request_cache_hit"] for r in saved["results"]] == [r["request_cache_hit"] for r in second] == [True] * 3


def test_changed_model_config_runs_again(make_tester, sent, tmp_path):
    make_tester(str(tmp_path / "run")).evaluate()
    (tmp_path / "model_configs" / "fake.json").write_text(json.dumps({**MODEL_CONFIG, "max_tokens": 20}))
    make_tester(str(tmp_path / "run")).evaluate()
    assert len(sent) == 6 and {r["max_tokens"] for r in sent} == {10, 20}
    (result_file,) = (tmp_path / "run").iterdir()
    assert json.loads(result_file.read_text())["model_config"]["max_tokens"] == 20
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_request_cache.py tests/test_async_evaluate.py -q`
Expected: both modules fail to import: `ImportError: cannot import name 'write_json_atomic'` and `ImportError: cannot import name 'result_file_name'`.

- [ ] **Step 3: Implement**

- `request_cache.py`: add `write_json_atomic` and use it in `RequestCache.put`.
- `async_evaluate.py`:
  - Add `result_file_name` and set `self.eval_name` from it.
  - Delete `get_hash`, the `prevent_duplicate` parameter and attribute, the `test_hash` line, the "Results already exist" early return and the directory scan.
  - Write the result with `write_json_atomic(results_path, outputs, indent=4)`, printing `Overwriting {results_path}` when the file exists.
  - Keep the model-config comment accurate: it is no longer "added after hashing". It is logged so the report can show the settings.
- `run_tests.py`: remove `--prevent_duplicate_tests` and `prevent_duplicate=...`.
- `README.md`:
  - "Collect the results": each test writes `<results_dir>/<model>_book_<test>_<haystack>_CL<length>.json`. Re-running a config overwrites it, and requests already in the request cache are not sent again, so re-running or resuming an interrupted run only pays for what is missing.
  - "Request cache": replace the `prevent_duplicate_tests` bullet with a note on result files from before this change. They have a timestamp in their name and are never overwritten. Delete them, or move them **outside** `evaluation/results/`, before re-running into the same folder. `gather_results` groups files by the `result_dir` recorded inside each file, not by where the file sits. Otherwise the report counts those tests twice, and it warns when it sees this.

- [ ] **Step 4: Run all tests**

Run: `make test`
Expected: 31 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/request_cache.py evaluation/async_evaluate.py evaluation/run_tests.py README.md tests/test_request_cache.py tests/test_async_evaluate.py
git commit -m "Overwrite each test's result file on re-run instead of skipping tests with existing results"
```

---

### Task 2: `gather_results.py` without `test_hash`, warning on repeated result files

**Files:**
- Modify: `evaluation/gather_results.py` (`get_summarized_result`, `summarize_runs`, `build_report`)
- Modify: `tests/test_gather_results.py`

**Interfaces:**
- Consumes: result files from Task 1 (no `test_hash`, still have `seed`).
- Produces:
  - `get_summarized_result` drops `test_hash` and `seed` independently (`data.pop(key, None)`). The CSV columns stay the same for old and new files.
  - `summarize_runs` summaries gain `"repeated_files": int`. It is the number of files beyond the first for each `(model_name, test_name, os.path.basename(haystack_path), context_length)` within a run.
  - When `repeated_files > 0`, `build_report` adds this to the run's **Warnings** (exact text): `f"{n} result file(s) repeat a model, test, haystack and context length already in this run (e.g. timestamped files from before re-runs overwrote results), so those requests are counted more than once. Delete the older files."`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_gather_results.py`)

```python
def test_summarized_result_drops_seed_without_test_hash():
    data = result_data()
    del data["test_hash"]
    s = get_summarized_result(data)
    assert "seed" not in s and "test_hash" not in s


def test_report_warns_about_repeated_result_files():
    loaded = [("old.json", result_data()), ("new.json", result_data())]
    report = build_report(summarize_runs(loaded), summarize_models(loaded), ["results"])
    assert "1 result file(s) repeat a model, test, haystack and context length already in this run" in report
    single = [("new.json", result_data())]
    assert "repeat a model, test, haystack" not in build_report(summarize_runs(single), summarize_models(single), ["results"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gather_results.py -q`
Expected: 2 failed (`seed` is still in the summary; the warning is missing).

- [ ] **Step 3: Implement** as in Interfaces. Update the module docstring's report line to mention the repeated-files warning.

- [ ] **Step 4: Run all tests, then the script on the real results**

Run: `make test && make gather-results && grep -c 'repeat a model, test, haystack' evaluation/results/report.md`
Expected: 33 passed. The same 12 result files are found. `grep -c` prints `0`, because the existing smoke results hold one file per test and haystack.

- [ ] **Step 5: Commit**

```bash
git add evaluation/gather_results.py tests/test_gather_results.py
git commit -m "Warn about repeated result files in the report and stop relying on test_hash"
```

---

## Final verification (live, paid, and deletes the old smoke results: ask the user first)

The 12 existing smoke result files have timestamped names and their prompts are not in the cache.

- [ ] With the user's OK, delete `evaluation/results/smoke_deepseek_v4.1_flash_1017_openrouter/`, or move it outside `evaluation/results/`.
- [ ] Run: `bash evaluation/run_tests.sh`
  Expected: 36 requests sent (about $0.0065). The folder holds 12 files named `…_rand_book_1_CL<length>.json`.
- [ ] Run: `bash evaluation/run_tests.sh` again.
  Expected: every test prints `requests: 0 sent, 6 from the request cache`. There are still 12 files, all with fresh modification times.
- [ ] Run: `make gather-results`. Expected: 36 requests, 36 from the request cache, current $0.0000, total about $0.0065, and no repeated-files warning.

## Out of scope

- Renaming or migrating old timestamped result files. They can't be served from the cache anyway, because they don't store the prompt.
- Speeding up re-runs by sharing the tokenizer and the encoded haystack across tests. Each test still loads its own today.

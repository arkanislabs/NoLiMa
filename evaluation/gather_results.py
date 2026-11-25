"""Gather NoLiMa result files into a CSV and a human-readable Markdown report.

CSV (one row per result file): the test settings, the logged model config (as
model_config.<field> columns) and
  result                  mean metric over the needle depths in that file
  n_requests, n_correct   requests in the file and how many scored 1
  n_cutoff                requests that hit max_tokens (finish_reason "length")
  total_input_tokens, total_output_tokens, total_reasoning_tokens
  n_request_cache_hits    requests answered from the local request cache instead of the API
  current_cost            USD of the requests sent to the API when the file was produced
  total_cost              USD of all requests, including those answered from the request cache
                          (costs are known only when the provider reports them, e.g. OpenRouter)
RAG result files (a top-level `rag` block) also get
  rag.<field>             the RAG settings, retriever and embedding config, flattened like model_config
  n_needle_retrieved      requests whose needle chunk was among the top_k retrieved chunks
  embedding_tokens        tokens sent to the embedding model for the file's requests
  embedding_cost          USD the provider reported for those embeddings (empty when not reported)

Report: per run (a parent_results_dir such as results/<run>/CL1K/), the score at
each context length, base score, threshold and effective length as defined in
the NoLiMa paper, plus per-test scores, cut-offs, current and total cost and the
model config, and, for RAG runs, the RAG settings and a retrieval table (needle
retrieval rate by the run's method and by hybrid, BM25 and dense rank, score on retrieved and missed requests, embedding
tokens and cost), and a warning when a run holds more than one file for the same
test, haystack and context length (those requests would be counted twice).

The request cache (evaluation/results/request_cache/) is skipped: its entries are
not result files.

Usage (from anywhere):
  python evaluation/gather_results.py                     # all of evaluation/results/
  python evaluation/gather_results.py RESULTS_DIR [...]   # specific result folders
  python evaluation/gather_results.py -o out.csv -r out.md
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from request_cache import current_cost, is_request_cache_dir

EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(EVALUATION_DIR, "results")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_RESULTS_DIR, "gathered_results.csv")
DEFAULT_REPORT_PATH = os.path.join(DEFAULT_RESULTS_DIR, "report.md")

# NoLiMa paper definitions
BASE_LENGTHS = (250, 500, 1000)   # base score: best of these per test, averaged over tests
THRESHOLD_RATIO = 0.85            # effective length: largest length scoring >= 85% of base
LOW_RATIO = 0.5                   # scores below 50% of base are italicized
FULL_RUN_REQUESTS = 5 * 58 * 26   # requests per context length in a full run (haystacks x tests x depths)
RANKINGS = ("hybrid", "bm25", "dense")   # the needle ranks a RAG result can log


# ---------------------------------------------------------------- loading

def load_result_files(results_dir: str) -> list:
    """Returns (path, data) for every result file under results_dir."""
    loaded = []
    if is_request_cache_dir(results_dir):
        return loaded
    for dirpath, dirnames, filenames in os.walk(results_dir):
        # The request cache can live inside results_dir; its entries are not result files
        dirnames[:] = [d for d in dirnames if not is_request_cache_dir(os.path.join(dirpath, d))]
        for filename in sorted(filenames):
            # Only result files; skips the gathered CSV and report saved in the same tree
            if not filename.endswith(".json"):
                continue
            # Comment this if you want to include direct results
            if "direct" in filename:
                continue
            result_path = os.path.join(dirpath, filename)
            with open(result_path, "r") as f:
                loaded.append((result_path, json.load(f)))
    return loaded


def run_name(data: dict) -> str:
    """Name of the run a result belongs to: the folder above CL<length>/<test_name>/."""
    result_dir = os.path.normpath(data.get("result_dir", ""))
    return os.path.basename(os.path.dirname(os.path.dirname(result_dir))) or "(unknown run)"


def sum_or_none(values: list):
    present = [v for v in values if v is not None]
    return sum(present) if present else None


# ---------------------------------------------------------------- CSV

def get_summarized_result(data: dict) -> dict:
    data = dict(data)
    results = data["results"]
    data["run"] = run_name(data)
    data["result"] = np.round(np.mean([res["metric"] for res in results]), decimals=4)
    data["n_requests"] = len(results)
    data["n_correct"] = sum(res["metric"] for res in results)
    data["n_cutoff"] = sum(res.get("finish_reason") == "length" for res in results)
    data["total_input_tokens"] = sum([res["prompt_tokens"] for res in results])
    data["total_output_tokens"] = sum([res["completion_tokens"] for res in results])
    data["total_reasoning_tokens"] = sum_or_none([res.get("reasoning_tokens") for res in results])
    data["n_request_cache_hits"] = sum(bool(res.get("request_cache_hit")) for res in results)
    data["current_cost"] = sum_or_none([current_cost(res) for res in results])
    data["total_cost"] = sum_or_none([res.get("cost") for res in results])
    for key, value in (data.pop("model_config", None) or {}).items():
        data[f"model_config.{key}"] = value
    data.pop("test_hash", None)   # only in files written before re-runs overwrote results
    data.pop("seed", None)
    rag = data.pop("rag", None)
    if rag is not None:
        rag_results = [res.get("rag") or {} for res in results]
        data["n_needle_retrieved"] = sum(bool(r.get("needle_retrieved")) for r in rag_results)
        data["embedding_tokens"] = sum((r.get("embedding") or {}).get("tokens") or 0 for r in rag_results)
        data["embedding_cost"] = sum_or_none([(r.get("embedding") or {}).get("cost") for r in rag_results])
        for key, value in rag.items():
            data[f"rag.{key}"] = value
    del data["result_dir"]
    del data["results"]
    return data


def clean_dict(d: dict) -> dict:
    """Remove newlines from string values so each result stays on one CSV line."""
    return {k: v.replace("\n", " ") if isinstance(v, str) else v for k, v in d.items()}


def write_csv(results: list, output_path: str) -> None:
    cleaned_results = [clean_dict(result) for result in results]
    # Union of keys in first-seen order: some tests have extra fields (e.g. "distractor")
    keys = list(dict.fromkeys(k for result in cleaned_results for k in result))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(cleaned_results)


# ---------------------------------------------------------------- aggregation

def summarize_runs(loaded: list) -> dict:
    """Aggregates request-level results per run. Returns {run: summary}."""
    by_run = defaultdict(list)
    for path, data in loaded:
        by_run[run_name(data)].append((path, data))

    summaries = {}
    for run, files in sorted(by_run.items()):
        length_stats = defaultdict(lambda: {"n": 0, "correct": 0, "cutoff": 0, "cache_hits": 0, "cost": [], "current_cost": [],
                                            "reasoning": [], "haystacks": set(), "tests": set()})
        rag_stats = defaultdict(lambda: {"n": 0, "chunks": 0, "context_tokens": 0, **{name + suffix: 0 for name in RANKINGS for suffix in ("", "_n")},
                                         "found": 0, "retrieved_correct": 0, "missed_correct": 0, "no_needle": 0,
                                         "embedding_tokens": 0, "embedding_costs": []})
        rag_settings = {}
        rag_files = 0
        test_scores = defaultdict(lambda: defaultdict(list))   # test -> length -> metrics
        configs = {}
        files_without_config = 0
        for path, data in files:
            length = data["context_length"]
            stats = length_stats[length]
            rag = data.get("rag")
            if rag is not None and "method" not in rag:
                rag = {**rag, "method": "hybrid"}   # files from before the method was configurable
            rag_settings[json.dumps(rag, sort_keys=True)] = rag
            rag_files += rag is not None
            stats["haystacks"].add(os.path.basename(data.get("haystack_path", "")))
            stats["tests"].add(data["test_name"])
            for res in data["results"]:
                stats["n"] += 1
                stats["correct"] += res["metric"]
                stats["cutoff"] += res.get("finish_reason") == "length"
                stats["cache_hits"] += bool(res.get("request_cache_hit"))
                stats["cost"].append(res.get("cost"))
                stats["current_cost"].append(current_cost(res))
                stats["reasoning"].append(res.get("reasoning_tokens"))
                test_scores[data["test_name"]][length].append(res["metric"])
                if rag is not None and res.get("rag"):
                    r, ranks, top_k = res["rag"], res["rag"].get("needle_rank") or {}, rag["top_k"]
                    rs = rag_stats[length]
                    rs["n"] += 1
                    rs["chunks"] += r["n_chunks"]
                    rs["context_tokens"] += r["context_tokens"]
                    for name in RANKINGS:
                        if name in ranks:   # a bm25 run has no hybrid or dense rank
                            rs[name + "_n"] += 1
                            rs[name] += ranks[name] is not None and ranks[name] <= top_k
                    rs["no_needle"] += all(rank is None for rank in ranks.values())
                    if r["needle_retrieved"]:
                        rs["found"] += 1
                        rs["retrieved_correct"] += res["metric"]
                    else:
                        rs["missed_correct"] += res["metric"]
                    embedding = r.get("embedding") or {}
                    rs["embedding_tokens"] += embedding.get("tokens") or 0
                    rs["embedding_costs"].append(embedding.get("cost"))
            if "model_config" in data:
                configs[json.dumps(data["model_config"], sort_keys=True)] = data["model_config"]
            else:
                files_without_config += 1

        # Files beyond the first for the same test cell; a re-run overwrites its file, so these are
        # leftovers such as timestamped files from before that, and they count requests twice
        cells = Counter((data["model_name"], data["test_name"], os.path.basename(data.get("haystack_path", "")), data["context_length"])
                        for _, data in files)
        repeated_files = sum(count - 1 for count in cells.values())

        lengths = sorted(length_stats)
        scores = {length: length_stats[length]["correct"] / length_stats[length]["n"] for length in lengths}

        # Base score: per test, best score among the base lengths; then the mean over tests
        best_per_test = [max(np.mean(per_length[length]) for length in BASE_LENGTHS if length in per_length)
                         for per_length in test_scores.values() if any(length in per_length for length in BASE_LENGTHS)]
        base = float(np.mean(best_per_test)) if best_per_test else None
        threshold = THRESHOLD_RATIO * base if base is not None else None
        effective = None
        if threshold is not None:
            passing = [length for length in lengths if scores[length] >= threshold]
            effective = format_length(max(passing)) if passing else f"<{format_length(lengths[0])}"

        summaries[run] = {
            "model_names": sorted({data["model_name"] for _, data in files}),
            "n_files": len(files),
            "lengths": lengths,
            "scores": scores,
            "length_stats": length_stats,
            "test_scores": test_scores,
            "base": base,
            "base_lengths_present": [length for length in BASE_LENGTHS if length in length_stats],
            "threshold": threshold,
            "effective": effective,
            "configs": list(configs.values()),
            "files_without_config": files_without_config,
            "repeated_files": repeated_files,
            "rag": next((r for r in rag_settings.values() if r is not None), None),
            "rag_stats": dict(sorted(rag_stats.items())),
            "n_rag_settings": len(rag_settings),
        }
    return summaries


def summarize_models(loaded: list) -> dict:
    """Request counts and cost per model_name, across all runs. Returns {model_name: summary}."""
    models = defaultdict(lambda: {"runs": set(), "n": 0, "cache_hits": 0, "costs": [], "current_costs": []})
    for _, data in loaded:
        model = models[data["model_name"]]
        model["runs"].add(run_name(data))
        model["n"] += len(data["results"])
        model["cache_hits"] += sum(bool(res.get("request_cache_hit")) for res in data["results"])
        model["costs"] += [res.get("cost") for res in data["results"]]
        model["current_costs"] += [current_cost(res) for res in data["results"]]
    return dict(sorted(models.items()))


# ---------------------------------------------------------------- formatting

def format_length(length: int) -> str:
    return f"{length // 1000}K" if length >= 1000 and length % 1000 == 0 else str(length)


def pct(value) -> str:
    return "--" if value is None else f"{100 * value:.1f}"


def styled_score(score: float, summary: dict) -> str:
    text = pct(score)
    if summary["threshold"] is not None and score >= summary["threshold"]:
        return f"<ins>{text}</ins>"
    if summary["base"] is not None and score < LOW_RATIO * summary["base"]:
        return f"*{text}*"
    return text


def format_cost(cost) -> str:
    if cost is None:
        return "n/a"
    return "<$0.0001" if 0 < cost < 0.0001 else f"${cost:.4f}"


def md_table(header: list, rows: list) -> list:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join([":---"] + [":---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def config_differences(configs: list) -> list:
    keys = sorted({k for config in configs for k in config})
    return [k for k in keys if len({json.dumps(config.get(k), sort_keys=True) for config in configs}) > 1]


def build_report(summaries: dict, models: dict, results_dirs: list) -> str:
    all_lengths = sorted({length for summary in summaries.values() for length in summary["lengths"]})
    total_requests = sum(stats["n"] for s in summaries.values() for stats in s["length_stats"].values())
    total_hits = sum(stats["cache_hits"] for s in summaries.values() for stats in s["length_stats"].values())
    total_cost = sum_or_none([sum_or_none(stats["cost"]) for s in summaries.values() for stats in s["length_stats"].values()])
    total_current = sum_or_none([sum_or_none(stats["current_cost"]) for s in summaries.values() for stats in s["length_stats"].values()])

    lines = [
        "# NoLiMa results report",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from {', '.join(f'`{d}`' for d in results_dirs)}: "
        f"{len(summaries)} run(s), {sum(s['n_files'] for s in summaries.values())} result files, "
        f"{total_requests:,} requests ({total_hits:,} from the request cache), "
        f"current cost {format_cost(total_current)}, total cost {format_cost(total_cost)}.",
        "",
        "## Summary",
        "",
    ]
    header = ["Run", "Model", "Base score (×0.85: Thr.)", "Effective length"] + [format_length(length) for length in all_lengths]
    rows = []
    for run, s in summaries.items():
        base = "--" if s["base"] is None else f"{pct(s['base'])} ({pct(s['threshold'])})"
        cells = [styled_score(s["scores"][length], s) if length in s["scores"] else "--" for length in all_lengths]
        rows.append([f"`{run}`", ", ".join(s["model_names"]), base, s["effective"] or "--"] + cells)
    lines += md_table(header, rows)
    lines += [
        "",
        "Scores are accuracy (%) over all requests at that context length. "
        "**Base score**: for each test, the best of its 250 / 500 / 1K scores, averaged over tests. "
        "**Effective length**: the largest context length scoring at least 85% of the base score (the threshold). "
        "Scores at or above the threshold are <ins>underlined</ins>; scores below 50% of the base score are *italicized*.",
        "",
        "## Cost by model",
        "",
    ]
    rows = []
    for model_name, m in models.items():
        missing = sum(cost is None for cost in m["costs"])
        note = "" if not missing else ("not reported" if missing == m["n"] else f"{missing:,} requests without a reported cost")
        rows.append([model_name, ", ".join(f"`{run}`" for run in sorted(m["runs"])), f"{m['n']:,}", f"{m['cache_hits']:,}",
                     format_cost(sum_or_none(m["current_costs"])), format_cost(sum_or_none(m["costs"])), note])
    lines += md_table(["Model", "Runs", "Requests", "From cache", "Current cost (USD)", "Total cost (USD)", "Note"], rows)
    lines += [
        "",
        "**Current cost**: requests sent to the API when the results were produced. "
        "**Total cost**: all requests, including those answered from the local request cache, at the price recorded when they were first sent. "
        "Costs are what the provider reported per request (e.g. OpenRouter's `usage.cost`).",
        "",
    ]

    for run, s in summaries.items():
        lines += [f"## Run `{run}`", ""]
        lines += [f"- **Model:** {', '.join(s['model_names'])}"]
        lines += [f"- **Result files:** {s['n_files']}"]
        n_requests = sum(stats["n"] for stats in s["length_stats"].values())
        n_hits = sum(stats["cache_hits"] for stats in s["length_stats"].values())
        run_current = sum_or_none([sum_or_none(stats["current_cost"]) for stats in s["length_stats"].values()])
        run_total = sum_or_none([sum_or_none(stats["cost"]) for stats in s["length_stats"].values()])
        lines += [f"- **Requests:** {n_requests:,} ({n_hits:,} from the request cache)"]
        lines += [f"- **Cost:** current {format_cost(run_current)}, total {format_cost(run_total)}"]
        if len(s["configs"]) == 1:
            config = s["configs"][0]
            settings = ", ".join(f"`{k}` = `{json.dumps(config[k])}`" for k in ("model", "max_tokens", "temperature", "top_p") if k in config)
            lines += [f"- **Model config:** {settings}"]
        elif s["configs"]:
            lines += [f"- **Model config:** {len(s['configs'])} different configs (see warnings and the full configs below)"]
        if s["rag"] is not None:
            rag = s["rag"]
            settings = [f"`method` = `{rag['method']}`"] + [f"`{k}` = {rag[k]}" for k in ("chunk_size", "chunk_overlap", "top_k")]
            settings.append(f"`embedding_model` = `{rag['embedding_model']}`" if rag["embedding_model"] else "`embedding_model` = none")
            settings.append("`retriever` = `" + ", ".join(f"{k}={v}" for k, v in rag["retriever"].items()) + "`")
            lines += [f"- **RAG:** {', '.join(settings)}"]
        lines += [""]

        warnings = []
        if s["n_rag_settings"] > 1:
            warnings.append(f"Results mix {s['n_rag_settings']} different RAG settings (no RAG counts as one); their scores are averaged together.")
        no_needle = sum(rs["no_needle"] for rs in s["rag_stats"].values())
        if no_needle:
            warnings.append(f"{no_needle} request(s): the needle was not found whole in any chunk, so retrieval counts it as missed.")
        if s["repeated_files"]:
            warnings.append(f"{s['repeated_files']} result file(s) repeat a model, test, haystack and context length already in this run "
                            "(e.g. timestamped files from before re-runs overwrote results), so those requests are counted more than once. "
                            "Delete the older files.")
        if s["files_without_config"]:
            warnings.append(f"{s['files_without_config']} of {s['n_files']} result files predate model-config logging, so their settings can't be checked.")
        if len(s["configs"]) > 1:
            warnings.append(f"Results mix {len(s['configs'])} different model configs (differing fields: {', '.join(f'`{k}`' for k in config_differences(s['configs']))}); their scores are averaged together.")
        missing_base = [format_length(length) for length in BASE_LENGTHS if length not in s["base_lengths_present"]]
        if s["base"] is None:
            warnings.append("No 250 / 500 / 1K results, so there is no base score, threshold or effective length.")
        elif missing_base:
            warnings.append(f"Base score is computed without {', '.join(missing_base)} results.")
        cutoffs = sum(stats["cutoff"] for stats in s["length_stats"].values())
        if cutoffs:
            warnings.append(f"{cutoffs} request(s) hit `max_tokens` (`finish_reason: length`) and were scored as wrong unless their partial answer matched.")
        small = [format_length(length) for length in s["lengths"] if s["length_stats"][length]["n"] < FULL_RUN_REQUESTS]
        if small:
            warnings.append(f"Fewer requests than a full run ({FULL_RUN_REQUESTS:,} per length) at {', '.join(small)}: treat those scores as rough.")
        if warnings:
            lines += ["**Warnings**", ""] + [f"- {w}" for w in warnings] + [""]

        lines += ["### Per context length", ""]
        header = ["Context length", "Score", "Correct / requests", "Tests", "Haystacks", "Cut off", "Avg reasoning tokens", "Current cost", "Total cost"]
        rows = []
        for length in s["lengths"]:
            stats = s["length_stats"][length]
            reasoning = [r for r in stats["reasoning"] if r is not None]
            rows.append([
                format_length(length), styled_score(s["scores"][length], s), f"{stats['correct']} / {stats['n']}",
                len(stats["tests"]), len(stats["haystacks"]), stats["cutoff"],
                f"{np.mean(reasoning):.0f}" if reasoning else "n/a",
                format_cost(sum_or_none(stats["current_cost"])), format_cost(sum_or_none(stats["cost"])),
            ])
        lines += md_table(header, rows) + [""]

        if s["rag_stats"]:
            lines += ["### Retrieval", "",
                      "Needle retrieved is the share (%) of requests whose needle chunk was among the `top_k` chunks the run's `method` retrieved; "
                      "Hybrid, BM25 alone and Dense alone are the same share by each ranking, whichever method picked the context "
                      "(-- where the run did not compute it: a `bm25` run embeds nothing). "
                      "Score is accuracy (%) on the requests whose needle was retrieved and on those where it was missed. "
                      "BM25 and dense use ordinal ranks that break ties by chunk position, so when every BM25 score is 0, BM25 alone counts a hit "
                      "exactly when the needle is in the first `top_k` chunks (chance level over uniform depths). "
                      "Embedding tokens and cost are what each result file recorded when it was written: texts served from the embedding cache count 0, "
                      "so a re-run shows 0 and $0.", ""]
            header = ["Context length", "Chunks", "Context tokens", "Needle retrieved", "Hybrid", "BM25 alone", "Dense alone",
                      "Score if retrieved", "Score if missed", "Embedding tokens sent", "Embedding cost"]
            rows = []
            for length, rs in s["rag_stats"].items():
                n, missed = rs["n"], rs["n"] - rs["found"]
                rows.append([
                    format_length(length), f"{rs['chunks'] / n:.1f}", f"{rs['context_tokens'] / n:.0f}",
                    pct(rs["found"] / n),
                    *(pct(rs[name] / rs[name + "_n"]) if rs[name + "_n"] else "--" for name in RANKINGS),
                    pct(rs["retrieved_correct"] / rs["found"]) if rs["found"] else "--",
                    pct(rs["missed_correct"] / missed) if missed else "--",
                    f"{rs['embedding_tokens']:,}", format_cost(sum_or_none(rs["embedding_costs"])),
                ])
            lines += md_table(header, rows) + [""]

        lines += ["### Per test", "", "Accuracy (%) of each question-needle test, over its haystacks and depths.", ""]
        header = ["Test"] + [format_length(length) for length in s["lengths"]]
        rows = [[f"`{test}`"] + [pct(np.mean(per_length[length])) if length in per_length else "--" for length in s["lengths"]]
                for test, per_length in sorted(s["test_scores"].items())]
        lines += md_table(header, rows) + [""]

        if s["configs"]:
            lines += ["<details><summary>Full model config</summary>", ""]
            for config in s["configs"]:
                lines += ["```json", json.dumps(config, indent=4), "```", ""]
            lines += ["</details>", ""]

    return "\n".join(lines)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Gather NoLiMa result files into a CSV and a Markdown report")
    parser.add_argument("results_dirs", nargs="*", default=[DEFAULT_RESULTS_DIR],
                        help=f"Result folders to search recursively (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_PATH,
                        help=f"CSV output path (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("-r", "--report", default=DEFAULT_REPORT_PATH,
                        help=f"Markdown report path (default: {DEFAULT_REPORT_PATH})")
    args = parser.parse_args()

    loaded = []
    for results_dir in args.results_dirs:
        if not os.path.isdir(results_dir):
            parser.error(f"Not a directory: {results_dir}")
        files = load_result_files(results_dir)
        loaded += files
        print(f"{results_dir}: {len(files)} result files")

    if not loaded:
        parser.error("No result files found")

    for path, data in loaded:
        data["result_name"] = path
    write_csv([get_summarized_result(data) for _, data in loaded], args.output)
    print(f"Wrote {len(loaded)} rows to {args.output}")

    report = build_report(summarize_runs(loaded), summarize_models(loaded), args.results_dirs)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        f.write(report + "\n")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()

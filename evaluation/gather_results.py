"""Gather NoLiMa result files into one CSV, with one row per result file.

Each row holds the test settings plus:
  result               mean metric over the needle depths in that file
  total_input_tokens   sum of prompt tokens
  total_output_tokens  sum of completion tokens

Usage (from anywhere):
  python evaluation/gather_results.py                     # all of evaluation/results/
  python evaluation/gather_results.py RESULTS_DIR [...]   # specific result folders
  python evaluation/gather_results.py -o out.csv          # custom output path
"""

import argparse
import csv
import json
import os

import numpy as np

EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(EVALUATION_DIR, "results")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_RESULTS_DIR, "gathered_results.csv")


def get_summarized_result(data: dict) -> dict:
    data["result"] = np.round(np.mean([res["metric"] for res in data["results"]]), decimals=4)
    data["total_input_tokens"] = sum([res["prompt_tokens"] for res in data["results"]])
    data["total_output_tokens"] = sum([res["completion_tokens"] for res in data["results"]])
    if "test_hash" in data:
        del data["test_hash"]
        del data["seed"]
    del data["result_dir"]
    del data["results"]
    return data


def gather_results(results_dir: str) -> list:
    results = []
    for dirpath, _, filenames in os.walk(results_dir):
        for filename in sorted(filenames):
            # Only result files; skips e.g. a gathered CSV saved in the same tree
            if not filename.endswith(".json"):
                continue
            # Comment this if you want to include direct results
            if "direct" in filename:
                continue
            result_path = os.path.join(dirpath, filename)
            with open(result_path, "r") as f:
                data = json.load(f)
            results.append(get_summarized_result(data))
            results[-1]["result_name"] = result_path
    return results


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Gather NoLiMa result files into one CSV")
    parser.add_argument("results_dirs", nargs="*", default=[DEFAULT_RESULTS_DIR],
                        help=f"Result folders to search recursively (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_PATH,
                        help=f"CSV output path (default: {DEFAULT_OUTPUT_PATH})")
    args = parser.parse_args()

    all_results = []
    for results_dir in args.results_dirs:
        if not os.path.isdir(results_dir):
            parser.error(f"Not a directory: {results_dir}")
        model_results = gather_results(results_dir)
        all_results += model_results
        print(f"{results_dir}: {len(model_results)} result files")

    if not all_results:
        parser.error("No result files found")

    write_csv(all_results, args.output)
    print(f"Wrote {len(all_results)} rows to {args.output}")


if __name__ == "__main__":
    main()

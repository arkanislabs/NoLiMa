# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import json
from typing import List, Optional, Union
from tqdm import tqdm
import time
from copy import copy

from jsonargparse import ArgumentParser, ActionConfigFile

from async_evaluate import NoLiMa_Tester
from embeddings import DEFAULT_EMBEDDING_CACHE_DIR
from rag import RAGConfig
from request_cache import DEFAULT_REQUEST_CACHE_DIR, cost_summary, current_cost
from data.book_haystack import BookHaystack

DEFAULT_TASK_TEMPLATE = "You will answer a question based on the following book snippet:\n\n{haystack}\n\nUse the information provided in the book snippet to answer the question. Your answer should be short and based on either explicitly stated facts or strong, logical inferences.\n\nQuestion: {question}\n\n Return only the final answer with no additional explanation or reasoning."

class EmbeddingProgress:
    """A second progress bar for RAG runs: retrievals done, embedding tokens sent and their cost, cache hits."""

    def __init__(self, total: int, desc: str):
        self.bar = tqdm(total=total, desc=desc, unit="req", dynamic_ncols=True, position=1)
        self.tokens = self.cache_hits = self.skipped = 0
        self.cost = 0.0
        self.cost_unknown = False
        self._show()

    def add(self, info: dict) -> None:
        """Count one request's retrieval log (the `rag` entry of a result)."""
        usage = info["embedding"]
        self.tokens += usage["tokens"]
        self.cache_hits += usage["cache_hits"]
        self.skipped += not info["ranked"]
        if usage["cost"] is None:
            self.cost_unknown = self.cost_unknown or usage["texts_sent"] > 0
        else:
            self.cost += usage["cost"]
        self.bar.update(1)
        self._show()

    def summary(self) -> str:
        cost = f"${self.cost:.4f}" + (" + unreported" if self.cost_unknown else "")
        return (f"{self.tokens:,} tokens sent, cost {cost}, {self.cache_hits:,} texts from the embedding cache, "
                f"{self.skipped:,} requests not embedded (every chunk retrieved)")

    def _show(self) -> None:
        self.bar.set_postfix_str(self.summary(), refresh=False)

    def close(self) -> None:
        self.bar.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="NoLiMa Multi-setup Tester")
    parser.add_argument('--config', action=ActionConfigFile, help='Path to a configuration YAML file.')
    parser.add_argument("--model_name", type=str, help="Name of the model to test")
    parser.add_argument("--model_configs_dir", type=str, help="Directory containing model configurations")
    parser.add_argument('--needle_set_path', type=str, help='Path to a file containing the needle tests configuration')
    parser.add_argument('--haystack_dir', type=str, help='Directory containing the haystack files')
    parser.add_argument("--task_template", type=str, help="Name of the model to test")
    parser.add_argument("--use_default_system_prompt", type=bool, help="Use default system prompt")
    parser.add_argument("--parent_results_dir", type=str, help="Parent directory to save results")
    parser.add_argument("--context_length", type=int, help="Context length for the needle placement")
    parser.add_argument("--document_depth_percent_min", type=float, default=0, help="Minimum document depth percentage")
    parser.add_argument("--document_depth_percent_max", type=float, default=100, help="Maximum document depth percentage")
    parser.add_argument("--document_depth_percent_intervals", type=int, default=35, help="Number of intervals between min and max depth")
    parser.add_argument("--shift", type=int, default=0, help="Shift applied to the beginning of the haystack")
    parser.add_argument("--static_depth", type=float, default=-1, help="Static depth for needle placement")
    parser.add_argument("--metric", type=str, default="EM", help="Evaluation metric")
    parser.add_argument("--log_placements_dir", type=str, default="", help="Directory to save needle placements for debugging")
    parser.add_argument("--base_seed", type=int, default=42, help="Base seed for random character selection")
    parser.add_argument("--test_names", type=Optional[List[str]], default=None, help="Run only these tests, named <exp_id>_<test_id>_<question_type> (e.g. 0401_T17_C02_onehop); default runs all")
    parser.add_argument("--haystack_files", type=Optional[List[str]], default=None, help="Run only these files from haystack_dir (e.g. rand_book_1.txt); default runs all")
    parser.add_argument("--request_cache_dir", type=str, default=DEFAULT_REQUEST_CACHE_DIR, help="Directory of the LLM request cache, shared by all runs")
    parser.add_argument("--use_request_cache", type=bool, default=True, help="Answer identical requests from the request cache instead of the API")
    parser.add_argument("--rag", type=Optional[RAGConfig], default=None, help="RAG baseline: answer from the top_k retrieved chunks of the haystack instead of the whole haystack")
    parser.add_argument("--embedding_cache_dir", type=str, default=DEFAULT_EMBEDDING_CACHE_DIR, help="Directory of the embedding cache, shared by all runs")
    parser.add_argument("--progress_label", type=str, default="", help="Prefix for the progress bar, e.g. the stage of a multi-run script ([2/5])")

    args = parser.instantiate_classes(parser.parse_args())

    with open(args.needle_set_path, "r") as file:
        needle_set = json.load(file)

    tests = []

    for exp_config in needle_set:
        system_prompt = exp_config["system_prompt"]
        exp_id = exp_config["id"]
        for question_type, question in exp_config["questions"].items():
            for test_id, test in exp_config["tests"].items():
                full_needle = "" + exp_config["needle"]
                input_args = test["input_args"]
                tests.append({
                    "test_name": exp_id+"_"+test_id+"_"+question_type,
                    "system_prompt": system_prompt,
                    "task_template": DEFAULT_TASK_TEMPLATE if "task_template" not in exp_config else exp_config["task_template"],
                    "gold_answers": test["gold_answers"] if "gold_answers" in test else "",
                    "seed": args.base_seed + int(exp_id[:4])
                })
                if "character_set" in exp_config:
                    tests[-1]["character_set"] = exp_config["character_set"]
                else:
                    tests[-1]["character_set"] = ""
                full_question = copy(question)
                full_distractor = None
                for arg_no, arg in enumerate(input_args):
                    arg_placeholder = "{"+str(arg_no+1)+"}"
                    if arg_placeholder in question:
                        full_question = question.replace(arg_placeholder, arg)
                    if arg_placeholder in full_needle:
                        full_needle = full_needle.replace(arg_placeholder, arg)
                    if "distractors" in exp_config and arg_placeholder in exp_config["distractors"][question_type]:
                        full_distractor = exp_config["distractors"][question_type].replace(arg_placeholder, arg)
                tests[-1]["needle"] = full_needle
                tests[-1]["retrieval_question"] = full_question
                tests[-1]["distractor"] = full_distractor

    if args.test_names is not None:
        unknown_tests = set(args.test_names) - {test["test_name"] for test in tests}
        if unknown_tests:
            raise ValueError(f"Unknown test_names (not in {args.needle_set_path}): {sorted(unknown_tests)}")
        tests = [test for test in tests if test["test_name"] in args.test_names]

    haystacks = os.listdir(args.haystack_dir)
    if args.haystack_files is not None:
        unknown_haystacks = set(args.haystack_files) - set(haystacks)
        if unknown_haystacks:
            raise ValueError(f"Unknown haystack_files (not in {args.haystack_dir}): {sorted(unknown_haystacks)}")
    selected_haystacks = [h for h in haystacks if args.haystack_files is None or h in args.haystack_files]
    n_runs = len(selected_haystacks) * len(tests)
    label = f"{args.progress_label} CL{args.context_length}".strip()
    print(f"{label}: {args.model_name}, {len(selected_haystacks)} haystacks x {len(tests)} tests x "
          f"{args.document_depth_percent_intervals} depths = {n_runs * args.document_depth_percent_intervals} requests"
          + (f" (RAG: top {args.rag.top_k} of {args.rag.chunk_size}-token chunks, overlap {args.rag.chunk_overlap})" if args.rag else ""))
    progress = tqdm(total=n_runs * args.document_depth_percent_intervals, desc=label, unit="req", dynamic_ncols=True)
    embedding_progress = EmbeddingProgress(progress.total, f"{label} embeddings") if args.rag else None
    run_no = 0
    spent = 0.0
    all_results = []
    # Built by the first test and reused by the rest: reloading the tokenizer and re-tokenizing each book
    # for every test took most of the time of a run whose requests were all cached
    api_connector = None
    for haystack_no, haystack in enumerate(haystacks):
        # Filter inside the loop so haystack_no, and therefore the seed, matches a full run
        if haystack not in selected_haystacks:
            continue
        haystack_path = os.path.join(args.haystack_dir, haystack)
        haystack_name = haystack.split(".")[0]
        book = BookHaystack(haystack_path)  # tokenized by its first test

        for test in tests:
            run_no += 1
            progress.set_postfix_str(f"test {run_no}/{n_runs} {haystack_name} {test['test_name']} | spent ${spent:.4f}")
            tester = NoLiMa_Tester(
                model_name=args.model_name,
                model_configs_dir=args.model_configs_dir,
                needle=test["needle"],
                haystack_path=haystack_path,
                results_dir=os.path.join(args.parent_results_dir, test["test_name"]),
                retrieval_question=test["retrieval_question"],
                gold_answers=json.dumps(test["gold_answers"]) if test["gold_answers"] != "" else "",
                character_set=json.dumps(test["character_set"]) if test["character_set"] != "" else "",
                system_prompt=test["system_prompt"],
                use_default_system_prompt=args.use_default_system_prompt,
                task_template=test["task_template"],
                context_length=args.context_length,
                document_depth_percent_min=args.document_depth_percent_min,
                document_depth_percent_max=args.document_depth_percent_max,
                document_depth_percent_intervals=args.document_depth_percent_intervals,
                shift=args.shift,
                static_depth=args.static_depth,
                metric=args.metric,
                log_placements_dir=args.log_placements_dir,
                test_name=test["test_name"],
                seed=test["seed"] + haystack_no,
                distractor=test["distractor"],
                request_cache_dir=args.request_cache_dir,
                use_request_cache=args.use_request_cache,
                api_connector=api_connector,
                haystack=book,
                rag=args.rag,
                embedding_cache_dir=args.embedding_cache_dir
            )
            api_connector = tester.api_connector

            results = tester.evaluate(progress=progress,
                                      on_retrieval=embedding_progress.add if embedding_progress is not None else None)
            spent += sum(c for c in map(current_cost, results) if c is not None)
            all_results += results
    progress.close()
    if embedding_progress is not None:
        embedding_progress.close()

    print(f"{label} done. {cost_summary(all_results)}")
    if embedding_progress is not None:
        print(f"{label} embeddings: {embedding_progress.summary()}")


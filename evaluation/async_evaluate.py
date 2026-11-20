# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import json
from typing import Union
from tqdm import tqdm
import asyncio

import numpy as np

from async_api_connector import APIConnector
from env_utils import load_env, expand_env_vars, redact_secrets
from request_cache import RequestCache, DEFAULT_REQUEST_CACHE_DIR, cost_summary, write_json_atomic

from jsonargparse import ArgumentParser, ActionConfigFile
from data.book_haystack import BookHaystack


def result_file_name(model_name: str, test_name: str, haystack_path: str, context_length: int) -> str:
    """
    Name (without .json) of a test's result file: the same test, haystack and context length always
    map to the same file, so a re-run overwrites it instead of adding a duplicate.
    """
    haystack_name = os.path.splitext(os.path.basename(haystack_path))[0]
    return "_".join(part for part in (model_name, "book", test_name, haystack_name, f"CL{context_length}") if part)


# Not test settings: the name, the outcome, and the model config (changing it and re-running is expected)
_NOT_SETTINGS = {"eval_name", "results", "model_config"}


def changed_settings(existing_path: str, outputs: dict) -> list:
    """
    Top-level settings (needle, haystack path, depths, seed, ...) that differ between an existing result
    file and the one about to replace it. The file name only encodes model, test, haystack file name and
    context length, so a different experiment pointed at the same folder would replace it silently.
    """
    try:
        with open(existing_path, "r") as file:
            existing = json.load(file)
    except (OSError, ValueError):
        return []
    new = json.loads(json.dumps(outputs))  # compare as stored (e.g. tuples become lists)
    keys = (set(existing) | set(new)) - _NOT_SETTINGS
    return sorted(key for key in keys if existing.get(key) != new.get(key))


class NoLiMa_Tester:
    def __init__(
        self,
        model_name: str,
        model_configs_dir: str,
        needle: str,
        haystack_path: str,
        results_dir: str,
        retrieval_question: str,
        system_prompt: str,
        use_default_system_prompt: bool,
        task_template: str,
        context_length: int,
        document_depth_percent_min = 0,
        document_depth_percent_max = 100,
        document_depth_percent_intervals = 35,
        seed: int = 42,
        gold_answers: str = "",
        character_set: str = "",
        shift: int = 0,
        static_depth: float = -1,
        test_name: str = "",
        log_placements_dir: str = "",
        metric: str = "EM",
        distractor: Union[str, None] = None,
        request_cache_dir: str = DEFAULT_REQUEST_CACHE_DIR,
        use_request_cache: bool = True
    ) -> None:
        self.model_name = model_name
        self.model_config_path = os.path.join(model_configs_dir, model_name + ".json")

        if not os.path.exists(self.model_config_path):
            raise FileNotFoundError(f"Model configuration for {model_name} not found in {model_configs_dir}")
        
        load_env()
        with open(self.model_config_path, "r") as file:
            raw_model_config = json.load(file)
        # Logged in each result file: ${VAR} references stay unexpanded and literal secrets are redacted
        self.logged_model_config = redact_secrets(raw_model_config)
        self.model_config = expand_env_vars(raw_model_config, source=self.model_config_path)

        # Identical requests (from this or any earlier run) are answered from the request cache
        self.api_connector = APIConnector(
            cache=RequestCache(request_cache_dir) if use_request_cache else None,
            **self.model_config
        )

        self.needle = needle
        if gold_answers != "":
            try:
                self.gold_answers = json.loads(gold_answers)
            except json.JSONDecodeError:
                self.gold_answers = gold_answers
        else:
            self.gold_answers = ""

        if character_set != "":
            try:
                self.character_set = json.loads(character_set)
            except json.JSONDecodeError:
                self.character_set = character_set
        else:
            self.character_set = ""

        self.system_prompt = system_prompt
        self.use_default_system_prompt = use_default_system_prompt
        self.task_template = task_template

        if not os.path.exists(haystack_path):
            raise FileNotFoundError(f"Haystack file not found at {haystack_path}")
        
        self.haystack_path = haystack_path

        self.haystack = BookHaystack(self.haystack_path)
        
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.retrieval_question = retrieval_question
        self.context_length = context_length
        self.document_depth_percent_min = document_depth_percent_min
        self.document_depth_percent_max = document_depth_percent_max
        self.document_depth_percent_intervals = document_depth_percent_intervals
        self.shift = shift
        self.static_depth = static_depth
        self.metric = metric
        self.seed = seed
        self.distractor = distractor
        
        self.log_placements = log_placements_dir != ""
        self.log_placements_dir = log_placements_dir

        self.test_name = test_name
        self.eval_name = result_file_name(model_name, test_name, haystack_path, context_length)

    
    def _evaluate_response(self, response: str, gold_answers = None) -> int:
        if gold_answers is None:
            gold_answers = self.gold_answers
        if self.metric == "EM":
            return int(response.strip() in gold_answers)
        elif self.metric == "contains":
            return int(any([gold_answer in response for gold_answer in gold_answers]))
        elif self.metric == "lastline_EM":
            return int(response.strip().split("\n")[-1] in gold_answers)
        elif self.metric == "lastline_contains":
            return int(any([gold_answer in response.strip().split("\n")[-1] for gold_answer in gold_answers]))
        else:
            raise ValueError(f"Invalid metric: {self.metric}")

    def evaluate(self) -> list:
        """
        Runs the test, (over)writes its result file and returns the per-request results. Requests
        already in the request cache are not sent again, so re-running a test only pays for what is missing.
        """
        np.random.seed(self.seed)
        outputs = {
            "eval_name": self.eval_name,
            "test_name": self.test_name,
            "model_name": self.model_name,
            "model_config_path": self.model_config_path,
            "retrieval_question": self.retrieval_question,
            "needle": self.needle,
            "gold_answers": self.gold_answers,
            "system_prompt": self.api_connector.SYSTEM_PROMPT if self.use_default_system_prompt else self.system_prompt,
            "use_default_system_prompt": self.use_default_system_prompt,
            "task_template": self.task_template,
            "haystack_path": self.haystack_path,
            "haystack_hash": self.haystack.get_hash(),
            "context_length": self.context_length,
            "character_set": self.character_set,
            "document_depth_percent_min": self.document_depth_percent_min,
            "document_depth_percent_max": self.document_depth_percent_max,
            "document_depth_percent_intervals": self.document_depth_percent_intervals,
            "shift": self.shift,
            "static_depth": self.static_depth,
            "metric": self.metric,
            "result_dir": self.results_dir,
            "log_placements_dir": self.log_placements_dir,
            "seed": self.seed,
            "results": []
        }
        if self.distractor is not None:
            outputs["distractor"] = self.distractor
        # Logged so the report can show the settings the results were produced with
        outputs["model_config"] = self.logged_model_config
        results_path = os.path.join(self.results_dir, f"{self.eval_name}.json")

        async_tasks = []
        for i in tqdm(np.linspace(self.document_depth_percent_min, self.document_depth_percent_max, self.document_depth_percent_intervals)):
            needle_depth = i / 100
            api_output = {}
            if "{CHAR}" in self.needle:
                if type(self.character_set) is not list:
                    raise ValueError("Character set not provided")
                selected_character = str(np.random.choice(self.character_set))
                api_output["selected_character"] = selected_character
                needle = self.needle.replace("{CHAR}", selected_character)
            else:
                needle = self.needle

            if "{CHAR}" in self.retrieval_question:
                retrieval_question = self.retrieval_question.replace("{CHAR}", selected_character)
            else:
                retrieval_question = self.retrieval_question

            placement_output = self.haystack.generate_w_needle_placement(
                needle=needle, 
                token_count_func=self.api_connector.token_count,
                encoding_func=self.api_connector.encode,
                decoding_func=self.api_connector.decode,
                context_length=self.context_length,
                depth=needle_depth,
                shift=self.shift,
                static_depth=self.static_depth,
                distractor=self.distractor
            )

            filled_template = self.task_template.format(haystack=placement_output["text"], question=retrieval_question)

            async_tasks.append(self.api_connector.generate_response(
                system_prompt=self.system_prompt,
                user_prompt=filled_template,
                max_tokens=self.model_config["max_tokens"], 
                temperature=self.model_config["temperature"],
                top_p=self.model_config["top_p"]
            ))

            api_output["placement_metadata"] = {k: v for k, v in placement_output.items() if k != "text"}

            outputs["results"].append(api_output)

            if self.log_placements:
                placement_log_path = os.path.join(self.log_placements_dir, f"{self.eval_name}_{str(np.round(i, 3))}.txt")
                with open(placement_log_path, "w") as file:
                    file.write(placement_output["text"])

        loop = asyncio.get_event_loop()

        # Let every request finish before failing on one: each response is cached as it arrives,
        # so a re-run only pays for the requests that failed
        responses = loop.run_until_complete(asyncio.gather(*async_tasks, return_exceptions=True))
        errors = [response for response in responses if isinstance(response, BaseException)]
        if errors:
            raise errors[0]

        for i in tqdm(range(len(responses))):
            outputs["results"][i]["metric"] = self._evaluate_response(responses[i]["response"], gold_answers=[outputs["results"][i]["selected_character"]]) if "{CHAR}" in self.needle else self._evaluate_response(responses[i]["response"])
            for k, v in responses[i].items():
                outputs["results"][i][k] = v
            
        # One file per model, test, haystack and context length; written atomically so a crash keeps the previous version
        if os.path.exists(results_path):
            changed = changed_settings(results_path, outputs)
            if changed:
                print(f"Warning: overwriting {results_path}, which was produced with different settings: {', '.join(changed)}. "
                      "Use a separate parent_results_dir to keep both.")
            else:
                print(f"Overwriting {results_path}")
        write_json_atomic(results_path, outputs, indent=4)
        
        print(f"Results saved at {results_path}")
        print(cost_summary(outputs["results"]))
        return outputs["results"]

    


if __name__ == "__main__":
    parser = ArgumentParser(description="NoLiMa Tester")
    parser.add_argument('--config', action=ActionConfigFile, help='Path to a configuration YAML file.')
    parser.add_argument("--model_name", type=str, help="Name of the model to test")
    parser.add_argument("--model_configs_dir", type=str, help="Directory containing model configurations")
    parser.add_argument("--needle", type=str, help="Needle to search in the haystack")
    parser.add_argument("--haystack_path", type=str, help="Path to the haystack file")
    parser.add_argument("--results_dir", type=str, help="Directory to save results")
    parser.add_argument("--retrieval_question", type=str, help="Question to retrieve the needle")
    parser.add_argument("--gold_answers", type=str, help="Gold answers to evaluate the response")
    parser.add_argument("--system_prompt", type=str, help="System prompt for the model")
    parser.add_argument("--use_default_system_prompt", type=bool, default=False, help="Use default system prompt")
    parser.add_argument("--task_template", type=str, help="Task template for the model")
    parser.add_argument("--system_prompt", type=str, help="System prompt for the model")
    parser.add_argument("--context_length", type=int, help="Context length for the needle placement")
    parser.add_argument("--document_depth_percent_min", type=int, default=0, help="Minimum document depth percentage")
    parser.add_argument("--document_depth_percent_max", type=int, default=100, help="Maximum document depth percentage")
    parser.add_argument("--document_depth_percent_intervals", type=int, default=35, help="Number of intervals between min and max depth")
    parser.add_argument("--static_depth", type=float, default=-1, help="Static depth for needle placement")
    parser.add_argument("--metric", type=str, default="EM", help="Evaluation metric")
    parser.add_argument("--log_placements_dir", type=str, default="", help="Directory to save needle placements for debugging")
    parser.add_argument("--test_name", type=str, default="", help="Name of the test")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random character selection")
    parser.add_argument("--distractor", type=str, default="", help="Distractor text placed in the haystack")
    parser.add_argument("--request_cache_dir", type=str, default=DEFAULT_REQUEST_CACHE_DIR, help="Directory of the LLM request cache, shared by all runs")
    parser.add_argument("--use_request_cache", type=bool, default=True, help="Answer identical requests from the request cache instead of the API")

    args = parser.parse_args()

    tester = NoLiMa_Tester(
        model_name=args.model_name,
        model_configs_dir=args.model_configs_dir,
        needle=args.needle,
        haystack_path=args.haystack_path,
        results_dir=args.results_dir,
        retrieval_question=args.retrieval_question,
        gold_answers=args.gold_answers,
        system_prompt=args.system_prompt,
        use_default_system_prompt=args.use_default_system_prompt,
        task_template=args.task_template,
        context_length=args.context_length,
        document_depth_percent_min=args.document_depth_percent_min,
        document_depth_percent_max=args.document_depth_percent_max,
        document_depth_percent_intervals=args.document_depth_percent_intervals,
        static_depth=args.static_depth,
        metric=args.metric,
        seed=args.seed,
        distractor=args.distractor if args.distractor != "" else None,
        request_cache_dir=args.request_cache_dir,
        use_request_cache=args.use_request_cache,
    )

    tester.evaluate()


# NoLiMa: Long-Context Evaluation Beyond Literal Matching

This repository contains the code and data associated with our ICML 2025 paper, "[NoLiMa: Long-Context Evaluation Beyond Literal Matching](https://arxiv.org/abs/2502.05167)".

## Abstract
> Recent large language models (LLMs) support long contexts ranging from 128K to 1M tokens. A popular method for evaluating these capabilities is the needle-in-a-haystack (NIAH) test, which involves retrieving a "needle" (relevant information) from a "haystack" (long irrelevant context). Extensions of this approach include increasing distractors, fact chaining, and in-context reasoning. However, in these benchmarks, models can exploit existing literal matches between the needle and haystack to simplify the task. To address this, we introduce **NoLiMa**, a benchmark extending NIAH with a carefully designed needle set, where questions and needles have **minimal lexical overlap, requiring models to infer latent associations to locate the needle within the haystack**. We evaluate 12 popular LLMs that claim to support contexts of at least 128K tokens. While they perform well in short contexts ($<$1K), performance degrades significantly as context length increases. At 32K, for instance, 10 models drop below 50\% of their strong short-length baselines. Even GPT-4o, one of the top-performing exceptions, experiences a reduction from an almost-perfect baseline of 99.3\% to 69.7\%. Our analysis suggests these declines stem from the increased difficulty the attention mechanism faces in longer contexts when literal matches are absent, making it harder to retrieve relevant information.

## Results
| Models               | Claimed Length | Effective Length | Base Score<br>(×0.85: Thr.) | 1K  | 2K  | 4K  | 8K  | 16K | 32K | 64K* | 128K* |
|----------------------|:-------------:|:---------------:|:-----------------------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4.1 🆕          | 1M            | 16K              | 97.0 (82.5)             | <ins>95.6</ins> | <ins>95.2</ins> | <ins>91.7</ins> | <ins>87.5</ins> | <ins>84.9</ins> | 79.8 | 69.7 | 64.7 |
| GPT-4o              | 128K          | 8K              | 99.3 (84.4)             | <ins>98.1</ins> | <ins>98.0</ins> | <ins>95.7</ins> | <ins>89.2</ins> | 81.6 | 69.7 | 62.4 | 56.0 |
| Llama 3.3 70B       | 128K          | 2K              | 97.3 (82.7)             | <ins>94.2</ins> | <ins>87.4</ins> | 81.5 | 72.1 | 59.5 | *42.7* | -- | -- |
| Llama 3.1 405B      | 128K          | 2K              | 94.7 (80.5)             | <ins>89.0</ins> | <ins>85.0</ins> | 74.5 | 60.1 | 48.4 | *38.0* | -- | -- |
| Llama 3.1 70B       | 128K          | 2K              | 94.5 (80.3)             | <ins>91.0</ins> | <ins>81.8</ins> | 71.2 | 62.7 | 51.8 | *43.2* | -- | -- |
| Gemini 1.5 Pro      | 2M            | 2K              | 92.6 (78.7)             | <ins>86.4</ins> | <ins>82.7</ins> | 75.4 | 63.9 | 55.5 | 48.2 | -- | -- |
| Jamba 1.5 Mini      | 256K          | <1K             | 92.4 (78.6)             | 76.3 | 74.1 | 70.8 | 62.2 | 52.7 | *43.6* | -- | -- |
| Command R+          | 128K          | <1K             | 90.9 (77.3)             | 77.0 | 73.5 | 66.3 | *39.5* | *21.3* | *7.4* | -- | -- |
| Llama 4 Maverick 🆕 | 1M            | 2K             | 90.1 (76.6)             | <ins>81.6</ins>  | <ins>78.3</ins> | 68.8 | 49.0 | *34.3* | *24.5* | -- | -- |
| Gemini 2.5 Flash (w/o T) 🆕 | 1M            | 2K             | 94.4 (80.2)             | <ins>90.1</ins> | <ins>86.1</ins> | 79.4 | 68.2 | 57.9 | 48.4 | -- | -- |
| Gemini 2.0 Flash 🆕 | 1M            | 4K             | 89.4 (76.0)             | <ins>87.7</ins> | <ins>87.5</ins> | <ins>77.9</ins> | 64.7 | 48.2 | *41.0* | *33.0* | *16.4* |
| Gemma 3 27B 🆕      | 128K          | <1K             | 88.6 (75.3)             | 73.3 | 65.6 | 48.1 | *32.7* | *20.2* | *9.5* | -- | -- |
| Mistral Large 2     | 128K          | 2K              | 87.9 (74.7)             | <ins>86.1</ins> | <ins>85.5</ins> | 73.3 | 51.5 | *32.6* | *18.7* | -- | -- |
| Claude 3.5 Sonnet   | 200K          | 4K              | 87.6 (74.4)             | <ins>85.4</ins> | <ins>84.0</ins> | <ins>77.6</ins> | 61.7 | 45.7 | *29.8* | -- | -- |
| Gemma 3 12B 🆕      | 128K          | 1K              | 87.4 (74.3)             | <ins>74.7</ins> | 61.8 | *39.9* | *27.4* | *16.8* | *7.3* | -- | -- |
| Gemini 1.5 Flash    | 1M            | <1K             | 84.7 (72.0)             | 68.6 | 61.6 | 51.0 | 44.4 | *35.5* | *28.6* | -- | -- |
| GPT-4o mini         | 128K          | <1K             | 84.9 (72.2)             | 67.7 | 58.2 | 44.1 | *32.6* | *20.6* | *13.7* | -- | -- |
| Llama 4 Scout 🆕    | 10M           | 1K              | 81.7 (69.4)             | <ins>72.3<ins> | 61.8 | 50.8 | *35.5* | *26.9* | *21.6* | -- | -- |
| GPT-4.1 Mini 🆕     | 1M            | <1K             | 80.9 (68.8)             | 66.7 | 62.8 | 58.7 | 51.9 | 46.2 | *38.8* | -- | -- |
| GPT-4.1 Nano 🆕     | 1M            | <1K             | 80.7 (68.6)             | 60.8 | 48.2 | *36.7* | *28.8* | *19.5* | *9.4* | -- | -- |
| Llama 3.1 8B        | 128K          | 1K              | 76.7 (65.2)             | <ins>65.7</ins> | 54.4 | 44.1 | *31.9* | *22.6* | *14.2* | -- | -- |
| Gemma 3 4B 🆕       | 128K          | <1K              | 73.6 (62.6)             | 50.3 | *35.3* | *16.4* | *7.5* | *2.3* | *0.9* | -- | -- |

This table presents the performance results of selected models on NOLIMA tests. The **base score** represents a model’s accuracy on the task at short contexts (250, 500, and 1K) and serves as a controlled reference to measure performance degradation at longer contexts. 
The **effective length** is defined as the longest context where a model maintains at least 85% of its base score. Scores above this threshold are <ins>underlined</ins>, while scores dropping below 50% of the base score are *italicized*.
Longer context evaluations (64K and 128K) use a reduced number of placements (11 instead of 26) and account for token limit constraints, particularly for GPT-4o.

#### ✨ Updates:

- [2025-07-17]: Added evaluation results on GPT-o3 and GPT-o4 Mini on NoLiMa-Hard in the reasoning models section.
- [2025-06-09]: Added support for external API providers (e.g. Fireworks, OpenRouter, ...) Added evaluation results on GPT-4.1 series, Gemini 2.5 Flash (w/o Thinking), and Llama 4 Maverick. 
Gemini 2.5 Pro and Gemini 2.5 Flash (w/ Thinking) results are included in the NoLiMa-Hard section. Added evaluation results up to 128K for GPT-4o, 4.1 and Gemini 2.0 Flash.
- [2025-04-10]: Added evaluation results on Gemma 3 models (4B/12B/27B), Gemini 2.0 Flash, and Llama 4 Scout.

### NoLiMa-Hard Results
| Models                | Base Score | 4K  | 8K  | 16K | 32K |
|-----------------------|:---------:|:---:|:---:|:---:|:---:|
| **Baselines for comparison (w/o CoT)**     |           |     |     |     |     |
| GPT-4.1              | 96.0       | 69.8 | 58.4 | 54.5 | *45.4* |
| GPT-4o               | 99.9       | 90.7 | 75.6 | 61.1 | *38.5* |
| Gemini 2.5 Flash (w/o Thinking) 🆕 | 87.5       | 47.2 | *23.5* | *13.4* | *9.8* |
| **Llama 3.3 70B**     |           |     |     |     |     |
| - w/o CoT            | 98.3       | 55.5 | *37.2* | *16.7* | *8.9* |
| - w/ CoT             | 97.1       | 73.0 | 51.2 | *31.8* | *10.1* |
| **Reasoning Models**  |           |     |     |     |     |
| GPT-o3 🆕               | 100.0      | 94.4 | 86.2 | 74.9 | 58.5 |
| Gemini 2.5 Pro 🆕    | 99.1       | 73.9 | 63.0 | 58.6 | 58.6 |
| GPT-o1               | 99.9       | 92.0 | 78.0 | 60.1 | *31.1* |
| DeepSeek R1-Distill-Llama-70B   | 99.9       | 91.4 | 75.5 | *49.4* | *20.7* |
| GPT-o3 Mini          | 98.8       | 52.8 | *36.9* | *25.5* | *18.9* |
| Gemini 2.5 Flash (w/ Thinking) 🆕  | 89.5      | 48.5 | *33.6* | *21.9* | *15.7* |
| GPT-o4 Mini 🆕          | 99.6     | 57.4 | *30.8* | *20.2* | *11.7* |


This table presents the performance results of selected reasoning models on **NoLiMa-Hard**, a subset of the original NoLiMa needle set containing the 10 most challenging question-needle pairs from previous evaluations. 
Scores dropping below 50% of the base score are in *italic*.


## Model Evaluation Instructions

Below are the general steps to evaluate models, whether serving them locally or using an API-based service.

---
### 1. Installing Requirements
Dependencies are managed with [uv](https://docs.astral.sh/uv/getting-started/installation/) (install it first). Then, from the repository root:
```bash
make install        # API-based models (OpenAI, Azure OpenAI, Gemini/Vertex AI, AWS Bedrock)
make install-local  # also installs vllm for locally served models (Linux + NVIDIA GPU)
source .venv/bin/activate
```
Both commands create a Python 3.12 environment in `.venv` from the pinned versions in `uv.lock`. `make install` keeps the environment in sync with the core dependencies, so running it after `make install-local` removes vllm.

### 2. Downloading Data
Download the NoLiMa dataset by running:
```bash
data/download_NoLiMa_data.sh
```
The needle set and haystack data will be downloaded to the `data` directory from our [HuggingFace Datasets 🤗](https://huggingface.co/datasets/amodaresi/NoLiMa) repository.

### 3A. Locally Served Models

1. **Start the model server (optional)**  
   - For example, to serve the Meta Llama 3.3 (70B) model across 8 GPUs:  
     ```bash
     evaluation/vllm_serve.sh --model_name meta-llama/Llama-3.3-70B-Instruct --num_gpus 8
     ```
   - This script uses a tensor parallel configuration by default. Modify it as needed.

2. **Create or modify a local model configuration**  
   - Use `llama_3.3_70b.json` in the `evaluation/model_configs` folder as a reference.
   Note that this configuration file is used in the evaluation script (not for the vllm serve).

### 3B. API-Based Models

- **Provide your API keys in a `.env` file**  
  - Copy the template and fill in the keys you need (`.env` is git-ignored):
    ```bash
    cp .env.example .env
    ```
  - Model configs reference keys as `${VAR_NAME}` (e.g. `"api_key": "${OPENROUTER_API_KEY}"`). The values are filled in from the environment when the config is loaded, and variables already exported in your shell take precedence over `.env`. A missing variable stops the run with an error that names it.
  - For OpenRouter, set `OPENROUTER_API_KEY` (and `HF_TOKEN` for gated Hugging Face tokenizers such as `meta-llama/*`), then use `evaluation/model_configs/deepseek_v4.1_flash_1017_openrouter.json` as a starting point.
  - To pin OpenRouter to a provider, add `"openrouter_provider"` to the model config. It is sent as OpenRouter's [`provider` routing object](https://openrouter.ai/docs/features/provider-routing), e.g. `{"only": ["openai"], "allow_fallbacks": false, "require_parameters": true}`, and is part of the request cache key. Each per-request result records the `provider` that served it and its `system_fingerprint`, so you can check the pin held. `gpt_4o_mini_openrouter.json` (`openai/gpt-4o-mini`) is pinned to OpenAI and `deepseek_v4.1_flash_1017_nothink_openrouter.json` to Cloudflare; their run configs are `evaluation/run_config/nolima_<model>_<length>.yaml`.
  - To control reasoning on OpenRouter, add `"openrouter_reasoning"` to the model config. It is sent as OpenRouter's [`reasoning` object](https://openrouter.ai/docs/use-cases/reasoning-tokens), e.g. `{"enabled": false}` to turn thinking off or `{"effort": "high"}`, and is part of the request cache key. `deepseek_v4.1_flash_1017_nothink_openrouter.json` runs DeepSeek V4.1 Flash (1017 snapshot) with thinking off, pinned to Cloudflare, with the paper's non-reasoning settings (192 max tokens, greedy decoding). Its run configs follow the same `nolima_<model>_<length>.yaml` pattern.
- **Create or modify a model configuration for your API-based service**  
  - For example, use the existing config templates in the `evaluation/model_configs` folder.  
  - Note that some APIs may require additional credentials or authentication (AWS, Google Auth, etc.).
  - For general API providers like Fireworks or OpenRouter, you will need to specify the `tokenizer_type` and `tokenizer_model` in the model config file (e.g. check `evaluation/model_configs/llama_4_scout_EXTERNAL_API.json` for an example).
  - Some models may require additional parameters, such as `thinking_budget` for Gemini 2.5 Flash, which can be specified in the model config file.

### 4. Common Steps for Both Approaches

1. **Prepare test configuration files**  
   - Add or modify configuration files in the `evaluation/run_config` directory.  
   - Ensure they reference the correct model config file from `evaluation/model_configs`.

2. **Run the evaluations**  
   ```bash
   cd evaluation/
   ./run_tests.sh
   ```
3. **Collect the results**
    - All outputs are automatically saved under the `parent_results_dir` of each run_config file: one file per test, haystack and context length, `<parent_results_dir>/<test>/<model>_book_<test>_<haystack>_CL<length>.json`.
    - Re-running a config overwrites these files. Requests already in the request cache (below) are not sent again, so re-running, or resuming an interrupted run, only pays for requests that aren't cached yet; a changed model config (e.g. `max_tokens`) runs the new settings. Two exceptions: with `use_request_cache: false` every request is sent again, and results produced before the request cache existed aren't in it, so re-running their config sends all of their requests again (use `test_names` / `haystack_files` to run only what is missing).
    - The file name only covers model, test, haystack file name and context length. Any other change (needle set, `haystack_dir`, depth settings, seed, metric) needs its own `parent_results_dir`; otherwise the new results replace the old ones, with a warning naming the settings that differ.
    - Each run prints its cost per test and in total: **current** is what the requests sent to the API cost, **total** also counts the requests answered from the request cache (at the price recorded when they were first sent).

#### Request cache
- Every LLM request is cached on disk, independently of the results, in `evaluation/results/request_cache/` (set `request_cache_dir` in a run config to move it). An identical request, from this run or any earlier one, is answered from the cache and not sent or paid for again.
- A request is identified by the SHA-256 of its *request record*: provider, endpoint, model and the exact call arguments (messages, `max_tokens`, `temperature`, `top_p`, seed, ...). API keys, timeouts, retry and tokenizer settings are not part of it.
- Each entry is one JSON file, `<model>/<hash[:2]>/<hash>.json`, holding the request record and the response, including its reported cost. Texts longer than 2,000 characters (the haystack prompt) are stored as a SHA-256 digest with the first and last 200 characters.
- Each per-request result records `request_hash` (the entry's file name) and `request_cache_hit`.
- To send requests again, delete their entries (or a whole model folder). To bypass the cache for a run, set `use_request_cache: false`.
- Responses with `finish_reason: "error"` are not cached, and when a request fails, the responses that did arrive are already cached, so a re-run only sends the failed requests.
- Result files from before re-runs overwrote results have a timestamp in their name and are never overwritten, and their prompts are not in the cache. Delete them, or move them outside `evaluation/results/`, before re-running into the same folder: `gather_results` groups files by the `result_dir` recorded inside each file, not by where the file sits, so leftover old files make the report count those tests twice (it warns when it sees this).
4. **Gathering the results**
    - Run `make gather-results` (or `python evaluation/gather_results.py [RESULTS_DIR ...]`) to gather the results from the output files into `evaluation/results/gathered_results.csv`, containing the accuracy of each test.
    - The same command writes `evaluation/results/report.md`: per run, the score at each context length, the base score, threshold and effective length (as defined above), per-test scores, cut-off answers, cost and the model config. Cost is shown as **current** (requests sent to the API when the results were produced) and **total** (also counting requests answered from the request cache); the CSV has the matching `current_cost`, `total_cost` and `n_request_cache_hits` columns. The request cache directory is skipped. Each result file also records the model config it was run with (`${VAR}` references are kept unexpanded and literal secrets are redacted).

### Additional Notes
You can find various needle sets (e.g., CoT-style, multiple choice, direct, distractor-included) in `data/needlesets`.
Adjust any paths or configurations as needed for your specific environment.

#### RAG baseline
A run config with a `rag:` block answers from the `top_k` most relevant chunks of the haystack instead of the whole haystack. Without the block, nothing changes. The block has four fields, shown here with their defaults:

```yaml
rag:
  chunk_size: 1000        # tokens per chunk
  chunk_overlap: 200      # tokens shared by neighbouring chunks
  top_k: 3                # chunks put in the prompt
  embedding_model: text_embedding_3_small_openrouter   # a config name in model_configs_dir
```

- Chunk sizes are counted in the generator model's tokens, the same unit as `context_length`.
- Retrieval is hybrid: Reciprocal Rank Fusion (k = 60) of BM25 (vanilla Okapi, `k1 = 1.2`, `b = 0.75`, the non-negative Lucene IDF) and dense similarity with `text-embedding-3-small` through OpenRouter, pinned to the OpenAI provider. The method is fixed in `evaluation/rag.py`, not configurable. The retrieved chunks are joined in document order and put into the unchanged prompt template.
- Embeddings are cached in SQLite at `evaluation/results/embedding_cache/<model slug>.sqlite`, one file per embedding model. The key is the SHA-256 of the model, endpoint, provider pin and text. Delete a model's `.sqlite` file to embed again. Set `embedding_cache_dir` in the run config to use another folder.
- OpenRouter reports the cost of embedding requests, so the end of a run prints the embedding tokens sent and their cost, apart from the generator's cost.
- Lengths up to `chunk_size + (top_k - 1) * (chunk_size - chunk_overlap)` (2,600 tokens with the defaults, so 250 to 2K) fit in `top_k` chunks. Their prompt equals the full-context prompt, so the generations come from the request cache. Nothing is embedded either: when a request has no more chunks than `top_k`, every chunk is retrieved whatever the scores, so the embedding request is skipped (`rag.ranked` is false in the request log).
- Result files record the retrieval settings under `rag`, including the method as `rag.retriever`. The report shows it on the RAG line and adds a Retrieval table per run: chunks per request, context tokens, how often the needle chunk is in the top `top_k` (hybrid, BM25 alone, dense alone), the score when it was and was not retrieved, and the embedding tokens and cost (embeddings sent when the result files were written; texts served from the embedding cache count 0). BM25 alone and dense alone break rank ties by chunk position, so when every BM25 score is 0, BM25 alone is a hit exactly when the needle is in the first `top_k` chunks, which is chance level over uniform depths.

Configs for DeepSeek V4.1 Flash are in `evaluation/run_config/nolima_deepseek_v4.1_flash_1017_nothink_openrouter_rag_c1000_k3_*.yaml`, and `smoke_test_rag_*` are small checks.

---

## Haystack Filtering Pipeline

To replicate our evaluation results, you can directly use the shuffled texts available in the `data/haystack/rand_shuffle` directory. If you prefer to generate your own shuffled texts or run the full processing pipeline from scratch, refer to the `data/README.md` file for more information.

## Cite
If you use the **NoLiMa** dataset, filtering pipeline, or code from this repository, please cite the [paper](https://arxiv.org/abs/2502.05167):

```bibtex
@inproceedings{modarressi2025nolima,
  title={NoLiMa: Long-Context Evaluation Beyond Literal Matching},
  author={Modarressi, Ali and Deilamsalehy, Hanieh and Dernoncourt, Franck and Bui, Trung and Rossi, Ryan A. and Yoon, Seunghyun and Schütze, Hinrich},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://arxiv.org/abs/2502.05167}
}
```

## License

The evaluation code and needle set data is licensed under the [Adobe Research License](LICENSE). The license prohibits commercial use and allows non-commercial research use. For details about the haystack data, please refer to the [data/haystack/LICENSES.md](https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/haystack/LICENSES.md) file.


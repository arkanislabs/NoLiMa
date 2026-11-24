# Experiment log

A hand-written log of NoLiMa experiments, newest first. Entries include excerpts from the generated `report.md`.

<!--
Suggested entry format:

## YYYY-MM-DD: <title>

**Research question:**

**Short answer:**

**TL;DR:**

- **Run:**
- **Commit:**
- **Model / provider:**
- **Config:**
- **Context lengths:**
- **Cost:**

<excerpt from report.md>

Notes:
-->

## 2026-11-20: Reproducing GPT-4o mini from the paper

**Research question:** Can we reproduce the benchmark exactly?

**Short answer:** No.

**TL;DR:** We can run the benchmark end to end with the paper's settings and data. The results don't match exactly, though: our 4K score is 1.4 points below the paper's, and the gap keeps growing with context length. Run-to-run noise explains little of it. The likely cause is on OpenAI's side, which we can't control.

- **Run:** `gpt_4o_mini_openrouter`, plus `gpt_4o_mini_openrouter_rerun1` (a 1K re-run the same evening to measure noise)
- **Commit:** `b7a7eb4`
- **Model / provider:** `openai/gpt-4o-mini` (unpinned alias) via OpenRouter, restricted to OpenAI (`only: ["openai"]`, no fallbacks).
- **Config:** the paper's settings: greedy decoding (`temperature` 0, `top_p` 1), `max_tokens` 192, `o200k_base` tokenizer
- **Context lengths:** 250-4K. Each length has 7,540 requests (5 haystacks × 58 tests × 26 placements). 8K to 32K were not run to reduce spending.
- **Cost:** \$7.02 + 1K re-run cost another \$0.96.

### Paper vs ours

| Source | Claimed length | Effective length | Base score (×0.85: Thr.) | 250 | 500 | 1K | 2K | 4K | 8K | 16K | 32K |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Paper, [Table 3](https://arxiv.org/html/2502.05167v3#A3) | 128K | ≪1K | 84.8 (72.1) | n/r | n/r | 67.7 | 58.2 | 44.2 | 32.6 | 20.6 | 13.7 |
| Ours, `report.md` | 128K | 500 (<1K by the paper's rule) | 85.3 (72.5) | 83.4 | 82.3 | 67.2 | 57.4 | 42.8 | – | – | – |

### Differences

- **Scores:** The gap is spread evenly across the one-hop and two-hop subsets (paper Tables 11 and 12), and no single test or category accounts for it.
- **Effective length (500 vs ≪1K):** this is a reporting difference, not a different result. `gather_results.py` lets 250 and 500 count as the effective length and uses `>=` for the threshold.
- **Model snapshot:** the paper pins `gpt-4o-mini-20240718` (Appendix B, Table 8). We used the unpinned alias, because our OpenRouter guardrail blocks the pinned slug.
- **Endpoint and date:** the paper's results date from early 2025. Ours were run in November 2026 through OpenRouter to OpenAI.

The data can't explain the gap. The paper describes the dataset thoroughly, and we used the official needle set and haystacks from Hugging Face, which haven't changed since 2025-02-07. The prompt template, seeding and needle placement code are unchanged from the authors' code. The scoring rule ("any returned answer containing the correct name") and the base score computation match §4.3. All requests ended with `finish_reason: stop`, so none were cut off.

### How much of the gap is LLM variance

To measure noise, we re-sent all 7,540 identical requests at 1K with the request cache off and compared them pair by pair with the original run:

| | |
|:---|:---|
| Accuracy | 67.16 → 67.31 (+0.15 points) |

That is +0.15 points of noise vs a 0.5-point gap to the paper at 1K.

This re-run ran on the same day on the same serving setups, so it measures sampling noise only, not drift over time.

### Possible reasons

1. **OpenAI's serving setup has changed since the paper's run (most likely).** About 1.5 years passed between the two runs. Even within a single run, our responses carry 35 different `system_fingerprint` values, and pinning a snapshot doesn't pin the serving setup. This fits the pattern we see: slightly better at short contexts, increasingly worse at long ones.
2. **A different endpoint.** The authors probably ran OpenAI models through Azure OpenAI. Their GPT-4o and GPT-4.1 configs in this repo (`model_configs/gpt_4o_20241120.json`, `model_configs/gpt_4.1_20250414.json`) use `api_provider: azure-openai`. We used OpenAI through OpenRouter.
3. **The unpinned alias instead of the pinned snapshot (unlikely).** `gpt-4o-mini-2024-07-18` is the only GPT-4o mini snapshot and OpenAI's default for the alias, so both probably serve the same weights. We couldn't test this, because our OpenRouter guardrail blocks the pinned slug.

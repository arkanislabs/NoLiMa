# Config and data paths are relative to evaluation/, so run from there regardless of the caller's cwd
cd "$(dirname "$0")" || exit 1
export PYTHONPATH="../"
# The HF tokenizer runs before the process forks; disable its threads to silence the fork warning
export TOKENIZERS_PARALLELISM=false
echo $$

python -u run_tests.py --config run_config/nolima_gpt_4o_mini_openrouter_250.yaml
python -u run_tests.py --config run_config/nolima_gpt_4o_mini_openrouter_500.yaml
python -u run_tests.py --config run_config/nolima_gpt_4o_mini_openrouter_1K.yaml
python -u run_tests.py --config run_config/nolima_gpt_4o_mini_openrouter_2K.yaml
python -u run_tests.py --config run_config/nolima_gpt_4o_mini_openrouter_4K.yaml

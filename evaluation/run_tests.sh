# Config and data paths are relative to evaluation/, so run from there regardless of the caller's cwd
cd "$(dirname "$0")" || exit 1
export PYTHONPATH="../"
# The HF tokenizer runs before the process forks; disable its threads to silence the fork warning
export TOKENIZERS_PARALLELISM=false
echo $$

set -- \
    run_config/nolima_deepseek_v4.1_flash_1017_nothink_openrouter_rag_c1000_k3_250.yaml \
    run_config/nolima_deepseek_v4.1_flash_1017_nothink_openrouter_rag_c1000_k3_500.yaml \
    run_config/nolima_deepseek_v4.1_flash_1017_nothink_openrouter_rag_c1000_k3_1K.yaml \
    run_config/nolima_deepseek_v4.1_flash_1017_nothink_openrouter_rag_c1000_k3_2K.yaml \
    run_config/nolima_deepseek_v4.1_flash_1017_nothink_openrouter_rag_c1000_k3_4K.yaml \
    run_config/nolima_deepseek_v4.1_flash_1017_nothink_openrouter_rag_c1000_k3_8K.yaml \
    run_config/nolima_deepseek_v4.1_flash_1017_nothink_openrouter_rag_c1000_k3_16K.yaml \
    run_config/nolima_deepseek_v4.1_flash_1017_nothink_openrouter_rag_c1000_k3_32K.yaml

stage=0
for config in "$@"; do
    stage=$((stage + 1))
    python -u run_tests.py --config "$config" --progress_label "[$stage/$#]"
done

# Config and data paths are relative to evaluation/, so run from there regardless of the caller's cwd
cd "$(dirname "$0")" || exit 1
export PYTHONPATH="../"
echo $$
# DeepSeek smoke tests: 6 requests per context length (1 haystack x 2 tests x 3 depths)
python -u run_tests.py --config run_config/smoke_test_deepseek_v4.1_flash_1017_250.yaml
python -u run_tests.py --config run_config/smoke_test_deepseek_v4.1_flash_1017_500.yaml
python -u run_tests.py --config run_config/smoke_test_deepseek_v4.1_flash_1017_1K.yaml
python -u run_tests.py --config run_config/smoke_test_deepseek_v4.1_flash_1017_2K.yaml
python -u run_tests.py --config run_config/smoke_test_deepseek_v4.1_flash_1017_4K.yaml
python -u run_tests.py --config run_config/smoke_test_deepseek_v4.1_flash_1017_8K.yaml

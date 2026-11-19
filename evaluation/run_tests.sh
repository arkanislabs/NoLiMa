# Config and data paths are relative to evaluation/, so run from there regardless of the caller's cwd
cd "$(dirname "$0")" || exit 1
export PYTHONPATH="../"
echo $$
python -u run_tests.py --config run_config/smoke_test_deepseek_v4.1_flash_1017.yaml

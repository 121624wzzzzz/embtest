#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/wz/projects/mypro/get_useful"
PYTHON_BIN="/home/wz/anaconda3/envs/wzall/bin/python"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/ijcai_clean/src"
export REPO_ROOT

{
  echo "===== task2 start: $(date '+%Y-%m-%d %H:%M:%S') ====="
  "${PYTHON_BIN}" ijcai_clean/scripts/run_task2_model_series.py --devices auto
  echo "===== task2 done:  $(date '+%Y-%m-%d %H:%M:%S') ====="
} >> "${LOG_DIR}/task2_cron.log" 2>&1

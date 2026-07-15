#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yanghan/AlphaCFG"
LOG_DIR="${PROJECT_ROOT}/outputs/cfg-sem-k/logs"
RUN_DIR="${PROJECT_ROOT}/outputs/cfg-sem-k/full_experiments"
LOG_FILE="${LOG_DIR}/paper_full_cuda1_treelstm.log"
PID_FILE="${LOG_DIR}/paper_full_cuda1_treelstm.pid"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"
cd "${PROJECT_ROOT}"

PYTHONUNBUFFERED=1 setsid "${PROJECT_ROOT}/.venv/bin/python" run_variant.py \
  --variant cfg-sem-k \
  --mode pool \
  --network treelstm \
  --seed 0 \
  --pool-capacity 10 \
  --max-expression-length 10 \
  --device cuda:1 \
  --run-tag paper-full-cuda1-treelstm \
  --log-dir "${RUN_DIR}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

echo "$!" > "${PID_FILE}"
cat "${PID_FILE}"

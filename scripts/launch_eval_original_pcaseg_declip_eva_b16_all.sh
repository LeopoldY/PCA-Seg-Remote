#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/eval_original_pcaseg_declip_eva_b16_all_4gpu.sh"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/pcaseg_original_declip_eva_b16_coco_4gpu_bs1_scaled}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_OUTPUT}/eval_all_official}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/eval_all.log}"
PID_FILE="${PID_FILE:-${OUTPUT_ROOT}/eval_all.pid}"

mkdir -p "${OUTPUT_ROOT}"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(tr -d '[:space:]' < "${PID_FILE}")"
  if [[ "${OLD_PID}" =~ ^[0-9]+$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Evaluation is already running (PID ${OLD_PID})."
    echo "Log: ${LOG_FILE}"
    exit 1
  fi
fi

nohup env TRAIN_OUTPUT="${TRAIN_OUTPUT}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
  bash "${EVAL_SCRIPT}" "$@" > "${LOG_FILE}" 2>&1 &
EVAL_PID=$!
printf '%s\n' "${EVAL_PID}" > "${PID_FILE}"

echo "Started all-dataset evaluation (PID ${EVAL_PID})."
echo "Log: ${LOG_FILE}"
echo "Summary: ${OUTPUT_ROOT}/summary.txt"

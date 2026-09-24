#!/usr/bin/env bash
set -euo pipefail

# Start the original PCA-Seg reproduction in the background and record its PID.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_original_pcaseg_declip_eva_b16_4gpu.sh"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/pcaseg_original_declip_eva_b16_coco_4gpu_bs1_scaled}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/launch.log}"
PID_FILE="${PID_FILE:-${OUTPUT_DIR}/train.pid}"

mkdir -p "${OUTPUT_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(tr -d '[:space:]' < "${PID_FILE}")"
  if [[ "${OLD_PID}" =~ ^[0-9]+$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Training is already running (PID ${OLD_PID})."
    echo "Log: ${LOG_FILE}"
    exit 1
  fi
fi

nohup env OUTPUT_DIR="${OUTPUT_DIR}" bash "${TRAIN_SCRIPT}" "$@" \
  > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" > "${PID_FILE}"

echo "Started original PCA-Seg training (PID ${TRAIN_PID})."
echo "Log: ${LOG_FILE}"
echo "Follow: tail -f '${LOG_FILE}'"
echo "Stop: kill ${TRAIN_PID}"

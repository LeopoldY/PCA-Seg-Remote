#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_original_pcaseg_declip_coco_2gpu_bs4.sh"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs}"

start_job() {
  local dataset="$1"
  local run_name="$2"
  local log_file="${LOG_ROOT}/${run_name}.log"
  local pid_file="${LOG_ROOT}/${run_name}.pid"

  if [[ -f "${pid_file}" ]]; then
    local old_pid
    old_pid="$(tr -d '[:space:]' < "${pid_file}")"
    if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "${dataset} is already running (PID ${old_pid})."
      echo "Log: ${log_file}"
      return
    fi
  fi

  nohup bash "${TRAIN_SCRIPT}" "${dataset}" > "${log_file}" 2>&1 &
  local train_pid=$!
  printf '%s\n' "${train_pid}" > "${pid_file}"
  echo "Started ${dataset} original PCA-Seg training (PID ${train_pid})."
  echo "Log: ${log_file}"
}

mkdir -p "${LOG_ROOT}"
start_job DLRSD original_pcaseg_dlrsd_declip_coco_2gpu_bs4
start_job iSAID original_pcaseg_isaid_declip_coco_2gpu_bs4

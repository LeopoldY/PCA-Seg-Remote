#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_dual_teacher_no_attr_declip_coco_2gpu_bs4.sh"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs}"
RESUME="${RESUME:-0}"

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "RESUME must be 0 or 1: ${RESUME}" >&2
  exit 2
fi

start_job() {
  local dataset="$1"
  local gpu_ids="$2"
  local run_name="$3"
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

  nohup env GPU_IDS="${gpu_ids}" RESUME="${RESUME}" \
    bash "${TRAIN_SCRIPT}" "${dataset}" > "${log_file}" 2>&1 &
  local train_pid=$!
  printf '%s\n' "${train_pid}" > "${pid_file}"
  echo "Started ${dataset} on GPUs ${gpu_ids} (PID ${train_pid})."
  echo "Log: ${log_file}"
}

mkdir -p "${LOG_ROOT}"
start_job DLRSD 0,1 dual_teacher_no_attr_dlrsd_declip_coco_2gpu_bs4
start_job iSAID 2,3 dual_teacher_no_attr_isaid_declip_coco_2gpu_bs4

echo "Both jobs dispatched. RESUME=${RESUME}"

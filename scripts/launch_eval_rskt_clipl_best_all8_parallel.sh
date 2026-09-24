#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DLRSD_GPU_IDS="${DLRSD_GPU_IDS:-0,1}"
ISAID_GPU_IDS="${ISAID_GPU_IDS:-2,3}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/rskt_clipl_eval_parallel}"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/eval_rskt_clipl_best_all8_2gpu.sh"

[[ "${DLRSD_GPU_IDS}" =~ ^[0-9]+,[0-9]+$ && "${ISAID_GPU_IDS}" =~ ^[0-9]+,[0-9]+$ ]] || {
  echo 'Each dataset requires two numeric GPU IDs, e.g. 0,1 and 2,3.' >&2; exit 2;
}
IFS=',' read -r -a ids <<< "${DLRSD_GPU_IDS},${ISAID_GPU_IDS}"
for ((i=0; i<4; i++)); do
  for ((j=i+1; j<4; j++)); do
    if ((10#${ids[i]} == 10#${ids[j]})); then
      echo 'The four GPU IDs must be distinct.' >&2; exit 2
    fi
  done
done

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  command -v tmux >/dev/null || { echo 'tmux is required.' >&2; exit 2; }
  for session in rskt_clipl_eval_dlrsd rskt_clipl_eval_isaid; do
    if tmux has-session -t "=${session}" 2>/dev/null; then
      echo "Session ${session} already exists; inspect it before launching again." >&2
      exit 2
    fi
  done
  mkdir -p "${LOG_ROOT}"
fi

start_job() {
  local dataset="$1" gpu_ids="$2" session="$3" tag="$4"
  local output="${PROJECT_ROOT}/output/clip_vitl_336_${tag}_attr64_dual_teacher_aff_residual_moe_2gpu_bs4/eval_best_miou_all8"
  local logfile="${LOG_ROOT}/${session}_$(date +%Y%m%d_%H%M%S).log"
  local command
  printf -v command 'set -o pipefail; env GPU_IDS=%q OUTPUT_ROOT=%q bash %q %q 2>&1 | tee %q' \
    "${gpu_ids}" "${output}" "${EVAL_SCRIPT}" "${dataset}" "${logfile}"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "${session}: ${command}"
  else
    # Set session options before dispatching evaluation; keep failures visible.
    tmux new-session -d -s "${session}" -c "${PROJECT_ROOT}"
    tmux set-option -t "${session}" remain-on-exit on
    local shell_command
    printf -v shell_command 'bash -c %q' "${command}"
    tmux send-keys -t "${session}" -l "${shell_command}"
    tmux send-keys -t "${session}" Enter
    echo "${dataset}: GPUs ${gpu_ids}; tmux attach -t ${session}"
    echo "Log: ${logfile}"
  fi
}

start_job DLRSD "${DLRSD_GPU_IDS}" rskt_clipl_eval_dlrsd dlrsd
start_job iSAID "${ISAID_GPU_IDS}" rskt_clipl_eval_isaid isaid

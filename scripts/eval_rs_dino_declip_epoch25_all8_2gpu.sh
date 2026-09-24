#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID}" >&2
  exit 2
fi

DATASET="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-/mnt/data6/yc/open-vocab/DeCLIP/logs/declip_eva_b16_rs_dino_dlrsd_isaid_full_v1/checkpoints/epoch_25.pt}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/evaluation_results/rs_dino_declip_epoch25_2gpu_bs4}"

case "${DATASET}" in
  DLRSD)
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr64_rs_dino_declip_epoch25_2gpu_bs4}"
    EVAL_OUTPUT="${EVAL_OUTPUT:-${TRAIN_OUTPUT}/evaluation_ovsisbench_all_2gpu}"
    RESULT_FILE="${RESULT_FILE:-${RESULT_ROOT}/DLRSD_checkpoint_all_8_datasets.txt}"
    ;;
  iSAID)
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr64_rs_dino_declip_epoch25_2gpu_bs4}"
    EVAL_OUTPUT="${EVAL_OUTPUT:-${TRAIN_OUTPUT}/evaluation_ovsisbench_all_2gpu}"
    RESULT_FILE="${RESULT_FILE:-${RESULT_ROOT}/iSAID_checkpoint_all_8_datasets.txt}"
    ;;
esac

exec env \
  DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT}" \
  TRAIN_OUTPUT="${TRAIN_OUTPUT}" \
  EVAL_OUTPUT="${EVAL_OUTPUT}" \
  RESULT_ROOT="${RESULT_ROOT}" \
  RESULT_FILE="${RESULT_FILE}" \
  bash "${PROJECT_ROOT}/scripts/eval_rs_dino_trained_2gpu.sh" "${DATASET}"

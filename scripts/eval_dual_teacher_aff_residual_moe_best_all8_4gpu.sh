#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID} [detectron2 config overrides ...]" >&2
  exit 2
fi

TRAIN_DATASET="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

case "${TRAIN_DATASET}" in
  DLRSD)
    CONFIG="${CONFIG:-configs/eva_vitb_384_dlrsd_attr64_dual_teacher_aff_residual_moe.yaml}"
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr64_dual_teacher_aff_residual_moe_2gpu_bs4}"
    ;;
  iSAID)
    CONFIG="${CONFIG:-configs/eva_vitb_384_isaid_attr64_dual_teacher_aff_residual_moe.yaml}"
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr64_dual_teacher_aff_residual_moe_2gpu_bs4}"
    ;;
esac

exec env \
  CONFIG="${CONFIG}" \
  TRAIN_OUTPUT="${TRAIN_OUTPUT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_OUTPUT}/eval_best_miou_all8}" \
  FEATURE_FUSION_TYPE=aff_residual_moe \
  bash scripts/eval_dual_teacher_best_all8_4gpu.sh \
    "${TRAIN_DATASET}" "$@"

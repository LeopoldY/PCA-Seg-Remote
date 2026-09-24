#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID} [detectron2 config overrides ...]" >&2
  exit 2
fi
DATASET="$1"
shift
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${DATASET}" in
  DLRSD) CONFIG_TAG=dlrsd; GPU_IDS="${GPU_IDS:-0,1}" ;;
  iSAID) CONFIG_TAG=isaid; GPU_IDS="${GPU_IDS:-2,3}" ;;
esac
CONFIG="${CONFIG:-configs/eva_vitb_384_${CONFIG_TAG}_attr64_dual_teacher_aff_residual_moe.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/eva_vitb_384_${CONFIG_TAG}_attr64_dual_teacher_aff_residual_moe_2gpu_bs4}"

echo "Fusion: AFF residual + four packed experts; attribute centers: 64"
exec env GPU_IDS="${GPU_IDS}" CONFIG="${CONFIG}" OUTPUT_DIR="${OUTPUT_DIR}" \
  bash "${PROJECT_ROOT}/scripts/train_dual_teacher_rskt_protocol_2gpu_bs4.sh" \
  "${DATASET}" MODEL.SEM_SEG_HEAD.FEATURE_FUSION.TYPE aff_residual_moe "$@"

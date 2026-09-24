#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID} [detectron2 config overrides ...]" >&2
  exit 2
fi

DATASET="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-/mnt/data6/yc/open-vocab/DeCLIP/logs/declip_eva_b16_rs_dino_dlrsd_isaid_full_v1/checkpoints/epoch_25.pt}"

case "${DATASET}" in
  DLRSD)
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr64_rs_dino_declip_epoch25_2gpu_bs4}"
    ;;
  iSAID)
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr64_rs_dino_declip_epoch25_2gpu_bs4}"
    ;;
esac

exec env \
  DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  bash "${PROJECT_ROOT}/scripts/train_rs_dino_rskt_protocol_2gpu_bs4.sh" \
    "${DATASET}" "$@"

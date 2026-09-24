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
  DLRSD) CONFIG_TAG=dlrsd ;;
  iSAID) CONFIG_TAG=isaid ;;
esac
CONFIG="${CONFIG:-configs/eva_vitb_384_${CONFIG_TAG}_attr64_rs_dino_aff_residual_moe.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/eva_vitb_384_${CONFIG_TAG}_attr64_rs_dino_aff_residual_moe_2gpu_bs4}"

# The shared launcher applies command-line defaults after YAML loading, so
# pass the fusion type explicitly as well as selecting the AFF configuration.
exec env CONFIG="${CONFIG}" OUTPUT_DIR="${OUTPUT_DIR}" \
  FEATURE_FUSION_TYPE=aff_residual_moe \
  bash "${PROJECT_ROOT}/scripts/train_rs_dino_rskt_protocol_2gpu_bs4.sh" \
  "${DATASET}" "$@"

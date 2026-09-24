#!/usr/bin/env bash
# Evaluate each ViT-L model on the existing OVSISBench all-8 protocol.
set -euo pipefail
if [[ $# -lt 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID} [detectron2 config overrides ...]" >&2
  exit 2
fi
TRAIN_DATASET="$1"
shift
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${TRAIN_DATASET}" in
  DLRSD) TAG=dlrsd ;;
  iSAID) TAG=isaid ;;
esac
export CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/clip_vitl_336_${TAG}_attr64_dual_teacher_aff_residual_moe.yaml}"
export TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/clip_vitl_336_${TAG}_attr64_dual_teacher_aff_residual_moe_2gpu_bs4}"
export CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/ViT-L-14-336px.pt}"
export ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/${TRAIN_DATASET}_train_desc_openai_l14_336_cluster_64_embedding_bank.pth}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_OUTPUT}/eval_best_miou_all8}"
# Shared runner selects the best validation mIoU checkpoint and evaluates
# DLRSD, iSAID, Potsdam, Vaihingen, UDD5, LoveDA, UAVid, VDD in sequence.
# Attribute fusion remains enabled; both distillation teachers are disabled.
exec bash "${PROJECT_ROOT}/scripts/eval_rskt_clip_best_all8_2gpu.sh" "${TRAIN_DATASET}" "$@"

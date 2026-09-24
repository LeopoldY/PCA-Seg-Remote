#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID} [config overrides ...]" >&2
  exit 2
fi
DATASET="$1"
shift
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
case "${DATASET}" in
  DLRSD) TAG=dlrsd ;;
  iSAID) TAG=isaid ;;
esac
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/data6/yc/open-vocab/RSKT-Seg/pretrained/ViT-B-16.pt}"
CONFIG="${CONFIG:-configs/clip_vitb_384_${TAG}_attr20_dual_teacher_aff_residual_moe.yaml}"
ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/${DATASET}_train_desc_remoteclip_b32_cluster_20_embedding_bank.pth}"
# Use the server's local launch time; set OUTPUT_DIR explicitly to resume a run.
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/clip_vitb_384_${TAG}_attr20_dual_teacher_aff_residual_moe_remoteclip_attr_content1_context0p1_2gpu_bs4_${RUN_TIMESTAMP}}"
echo "Student: RSKT-Seg OpenAI CLIP ViT-B/16; 384px, four directions"
echo "Attribute bank: RemoteCLIP text space; fusion: AFFResidualMoE"
# The common launcher calls the student checkpoint variable DECLIP_CHECKPOINT;
# here it explicitly points to OpenAI CLIP, not an EVA checkpoint.
exec env CONFIG="${CONFIG}" DECLIP_CHECKPOINT="${CLIP_CHECKPOINT}" \
  ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE}" OUTPUT_DIR="${OUTPUT_DIR}" \
  bash scripts/train_dual_teacher_rskt_protocol_2gpu_bs4.sh "${DATASET}" \
  MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED ViT-B/16 \
  MODEL.SEM_SEG_HEAD.FEATURE_FUSION.TYPE aff_residual_moe "$@"

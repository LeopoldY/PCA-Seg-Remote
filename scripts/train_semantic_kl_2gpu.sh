#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID} [detectron2 config overrides ...]" >&2
  exit 2
fi
DATASET="$1"
shift
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
case "${DATASET}" in
  DLRSD) TAG=dlrsd; DEFAULT_GPUS=2,3 ;;
  iSAID) TAG=isaid; DEFAULT_GPUS=2,3 ;;
esac
GPU_IDS="${GPU_IDS:-${DEFAULT_GPUS}}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ ${#GPU_ARRAY[@]} -ne 2 || ! "${GPU_IDS}" =~ ^[0-9]+,[0-9]+$ || "${GPU_ARRAY[0]}" == "${GPU_ARRAY[1]}" ]]; then
  echo "GPU_IDS must contain two distinct GPU ids: ${GPU_IDS}" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
CONFIG="${CONFIG:-configs/clip_vitb_384_${TAG}_attr64_semantic_kl.yaml}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/data6/yc/open-vocab/RSKT-Seg/pretrained/ViT-B-16.pt}"
RS_DINO_CHECKPOINT="${RS_DINO_CHECKPOINT:-${PROJECT_ROOT}/pretrained/RSIB.pth}"
REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/RemoteCLIP-ViT-B-32.pt}"
ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/${DATASET}_train_desc_openai_b16_cluster_64_embedding_bank.pth}"
RESUME="${RESUME:-0}"
if [[ "${RESUME}" == "1" && -z "${OUTPUT_DIR:-}" ]]; then
  echo "RESUME=1 requires an explicit OUTPUT_DIR for the existing run" >&2
  exit 2
fi
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/clip_vitb_384_${TAG}_openai_attr64_semantic_kl_2gpu_bs4_${RUN_TIMESTAMP}}"
for path in "${PYTHON_BIN}" "${DATASET_ROOT}" "${CONFIG}" "${CLIP_CHECKPOINT}" "${RS_DINO_CHECKPOINT}" "${REMOTECLIP_CHECKPOINT}" "${ATTRIBUTE_DATABASE}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done
COMMAND=("${PYTHON_BIN}" train_net.py --config-file "${CONFIG}" --num-gpus 2 --dist-url auto)
if [[ "${RESUME}" == "1" ]]; then COMMAND+=(--resume); fi
COMMAND+=(OUTPUT_DIR "${OUTPUT_DIR}"
  MODEL.SEM_SEG_HEAD.CACHE_DIR "${CLIP_CHECKPOINT}"
  MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.WEIGHTS "${RS_DINO_CHECKPOINT}"
  MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.WEIGHTS "${REMOTECLIP_CHECKPOINT}"
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTRIBUTE_DATABASE}"
  "$@")
echo "Dataset=${DATASET}; GPUs=${GPU_IDS}; student=OpenAI CLIP ViT-B/16; config=${CONFIG}"
echo "Attributes=${ATTRIBUTE_DATABASE}"
echo "Output=${OUTPUT_DIR}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'OVSISBENCH_DATASETS=%q DETECTRON2_DATASETS=%q CUDA_VISIBLE_DEVICES=%q ' "${DATASET_ROOT}" "${DATASET_ROOT}" "${GPU_IDS}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
if [[ "${RESUME}" != "1" && -e "${OUTPUT_DIR}/last_checkpoint" ]]; then
  echo "Existing checkpoint in ${OUTPUT_DIR}; use a new OUTPUT_DIR or RESUME=1" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
export OVSISBENCH_DATASETS="${DATASET_ROOT}" DETECTRON2_DATASETS="${DATASET_ROOT}" CUDA_VISIBLE_DEVICES="${GPU_IDS}"
exec "${COMMAND[@]}"

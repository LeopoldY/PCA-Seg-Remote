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

DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"
RESUME="${RESUME:-0}"

case "${DATASET}" in
  DLRSD)
    GPU_IDS="${GPU_IDS:-0,1}"
    CONFIG="${CONFIG:-configs/eva_vitb_384_dlrsd_original_pcaseg_declip_coco.yaml}"
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/original_pcaseg_dlrsd_declip_coco_2gpu_bs4}"
    ;;
  iSAID)
    GPU_IDS="${GPU_IDS:-2,3}"
    CONFIG="${CONFIG:-configs/eva_vitb_384_isaid_original_pcaseg_declip_coco.yaml}"
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/original_pcaseg_isaid_declip_coco_2gpu_bs4}"
    ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 2 ]]; then
  echo "GPU_IDS must contain exactly two comma-separated GPU ids: ${GPU_IDS}" >&2
  exit 2
fi
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "RESUME must be 0 or 1: ${RESUME}" >&2
  exit 2
fi

for path in "${PYTHON_BIN}" "${CONFIG}" "${DECLIP_CHECKPOINT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file does not exist: ${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
LAUNCH_ARGS=(--config-file "${CONFIG}" --num-gpus 2 --dist-url auto)
if [[ "${RESUME}" == "1" ]]; then
  LAUNCH_ARGS+=(--resume)
fi

echo "Model: original PCA-Seg EVA02-CLIP-B/16"
echo "Dataset: ${DATASET}"
echo "GPUs: ${GPU_IDS}; batch: 4/GPU, 8 global"
echo "Attribute enhancement: disabled"
echo "RS-DINO: disabled"
echo "CLIP initialization: COCO-trained DeCLIP (${DECLIP_CHECKPOINT})"
echo "Training: 30000 iterations, AdamW, LR 0.0002, cosine schedule"
echo "Resume: ${RESUME}"
echo "Output: ${OUTPUT_DIR}"

OVSISBENCH_DATASETS="${DATASET_ROOT}" \
DETECTRON2_DATASETS="${DATASET_ROOT}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
"${PYTHON_BIN}" train_net.py \
  "${LAUNCH_ARGS[@]}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  MODEL.SEM_SEG_HEAD.CACHE_DIR "${DECLIP_CHECKPOINT}" \
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED False \
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "" \
  MODEL.SEM_SEG_HEAD.RS_DINO.ENABLED False \
  MODEL.SEM_SEG_HEAD.RS_DINO.WEIGHTS "" \
  SOLVER.IMS_PER_BATCH 8 \
  SOLVER.GRAD_ACCUM_STEPS 1 \
  "$@"

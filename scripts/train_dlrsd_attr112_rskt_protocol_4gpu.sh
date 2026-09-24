#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-1,2,3,4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
CONFIG="${CONFIG:-configs/eva_vitb_384_dlrsd_attr112_rskt_protocol.yaml}"
ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/rskt_seg_train_desc_eva02_clip_b16_original_cluster_112_embedding_bank.pth}"
EVA_CHECKPOINT="${EVA_CHECKPOINT:-${PROJECT_ROOT}/pretrained/EVA02_CLIP_B_psz16_s8B.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr112_rskt_protocol_4gpu_bs2}"
RESUME="${RESUME:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU ids: ${GPU_IDS}" >&2
  exit 2
fi

for path in "${PYTHON_BIN}" "${CONFIG}" "${ATTRIBUTE_DATABASE}" "${EVA_CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
LAUNCH_ARGS=(--config-file "${CONFIG}" --num-gpus 4 --dist-url auto)
if [[ "${RESUME}" == "1" ]]; then
  LAUNCH_ARGS+=(--resume)
fi

echo "Dataset: DLRSD (17 native classes)"
echo "GPUs: ${GPU_IDS}"
echo "Global batch: 8 (two images per GPU; no gradient accumulation)"
echo "RSKT-Seg protocol: AdamW, LR 0.0002, cosine schedule, 30000 iterations"
echo "Output: ${OUTPUT_DIR}"

OVSISBENCH_DATASETS="${DATASET_ROOT}" \
DETECTRON2_DATASETS="${DATASET_ROOT}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
"${PYTHON_BIN}" train_net.py \
  "${LAUNCH_ARGS[@]}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  MODEL.SEM_SEG_HEAD.CACHE_DIR "${EVA_CHECKPOINT}" \
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED True \
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTRIBUTE_DATABASE}" \
  "$@"

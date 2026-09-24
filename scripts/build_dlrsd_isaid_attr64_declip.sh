#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
GPU_ID="${GPU_ID:-0}"
RAW_DECLIP_CHECKPOINT="${RAW_DECLIP_CHECKPOINT:-/mnt/data6/yc/open-vocab/DeCLIP/logs/declip_eva_b16_dinov2b_coco_full_retry1/checkpoints/epoch_latest.pt}"
DECLIP_MODEL_CHECKPOINT="${DECLIP_MODEL_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"
DLRSD_DESCRIPTORS="${DLRSD_DESCRIPTORS:-${PROJECT_ROOT}/attributes_text/DLRSD_train_descriptors.json}"
ISAID_DESCRIPTORS="${ISAID_DESCRIPTORS:-${PROJECT_ROOT}/attributes_text/iSAID_train_descriptors.json}"
DLRSD_DATABASE="${DLRSD_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/DLRSD_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"
ISAID_DATABASE="${ISAID_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/iSAID_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"

for path in "${PYTHON_BIN}" "${RAW_DECLIP_CHECKPOINT}" "${DLRSD_DESCRIPTORS}" "${ISAID_DESCRIPTORS}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 2
  fi
done

if [[ ! -f "${DECLIP_MODEL_CHECKPOINT}" ]]; then
  "${PYTHON_BIN}" tools/export_declip_eva_checkpoint.py \
    --input "${RAW_DECLIP_CHECKPOINT}" \
    --output "${DECLIP_MODEL_CHECKPOINT}"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" tools/build_attribute_database.py \
  --descriptors-json "${DLRSD_DESCRIPTORS}" \
  --output "${DLRSD_DATABASE}" \
  --model-name EVA02-CLIP-B-16 \
  --checkpoint "${DECLIP_MODEL_CHECKPOINT}" \
  --backend eva \
  --device cuda:0 \
  --batch-size 128 \
  --num-clusters 64

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" tools/build_attribute_database.py \
  --descriptors-json "${ISAID_DESCRIPTORS}" \
  --output "${ISAID_DATABASE}" \
  --model-name EVA02-CLIP-B-16 \
  --checkpoint "${DECLIP_MODEL_CHECKPOINT}" \
  --backend eva \
  --device cuda:0 \
  --batch-size 128 \
  --num-clusters 64

echo "DLRSD database: ${DLRSD_DATABASE}"
echo "iSAID database: ${ISAID_DATABASE}"

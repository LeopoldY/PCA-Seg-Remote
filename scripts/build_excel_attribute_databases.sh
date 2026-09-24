#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -z "${PYTHON_BIN:-}" && -x /opt/miniconda3/envs/yc_d2/bin/python ]]; then
  PYTHON_BIN=/opt/miniconda3/envs/yc_d2/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
MODEL_NAME="${MODEL_NAME:-EVA02-CLIP-B-16}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/pretrained/EVA02_CLIP_B_psz16_s8B.pt}"
DEVICE="${DEVICE:-cuda}"
ATTR_DIR="${PROJECT_ROOT}/attributes_text/excel"

"${PYTHON_BIN}" tools/build_attribute_database.py \
  --descriptors-json "${ATTR_DIR}/descriptors_pascal_voc_gpt4.0_cluster_a_photo_of4.json" \
  --output "${ATTR_DIR}/pascal_voc_desc_eva02_clip_b16_gpt4.0_cluster_112_embedding_bank.pth" \
  --model-name "${MODEL_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --backend eva \
  --device "${DEVICE}" \
  --num-clusters 112

"${PYTHON_BIN}" tools/build_attribute_database.py \
  --descriptors-json "${ATTR_DIR}/descriptors_ms_coco_gpt4.0_cluster_a_photo_of4.json" \
  --output "${ATTR_DIR}/ms_coco_desc_eva02_clip_b16_gpt4.0_cluster_224_embedding_bank.pth" \
  --model-name "${MODEL_NAME}" \
  --checkpoint "${CHECKPOINT}" \
  --backend eva \
  --device "${DEVICE}" \
  --num-clusters 224

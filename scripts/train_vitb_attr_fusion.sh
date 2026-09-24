#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

NUM_GPUS="${1:-8}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/output/eva_vitb_384_excel_tse}"
shift $(( $# >= 2 ? 2 : $# ))

CONFIG="${CONFIG:-configs/eva_vitb_384_attr_fusion.yaml}"
ATTR_DATABASE="${ATTR_DATABASE:-${PROJECT_ROOT}/attributes_text/excel/ms_coco_desc_eva02_clip_b16_gpt4.0_cluster_224_embedding_bank.pth}"
OPENCLIP_PRETRAINED="${OPENCLIP_PRETRAINED:-${PROJECT_ROOT}/pretrained/EVA02_CLIP_B_psz16_s8B.pt}"
if [[ -z "${PYTHON_BIN:-}" && -x /opt/miniconda3/envs/yc_d2/bin/python ]]; then
  PYTHON_BIN=/opt/miniconda3/envs/yc_d2/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" train_net.py \
  --config-file "${CONFIG}" \
  --num-gpus "${NUM_GPUS}" \
  --dist-url auto \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  MODEL.SEM_SEG_HEAD.CACHE_DIR "${OPENCLIP_PRETRAINED}" \
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTR_DATABASE}" \
  "$@"

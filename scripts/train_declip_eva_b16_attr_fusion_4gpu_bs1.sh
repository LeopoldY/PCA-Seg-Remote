#!/usr/bin/env bash
set -euo pipefail

# Train PCA-Seg on COCO-Stuff with the completed DeCLIP-encoded ExCEL bank.
# Physical GPUs 1,2,3,4; one image/GPU; total batch 4.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/declip_eva_b16_dinov2b_coco_full_retry1_excel_tse_4gpu_bs1_scaled}"
DECLIP_WEIGHTS="${DECLIP_WEIGHTS:-${OUTPUT_DIR}/declip_eva_b16_epoch6_state_dict.pt}"
ATTR_DATABASE="${ATTR_DATABASE:-${OUTPUT_DIR}/ms_coco_desc_declip_eva_b16_cluster_224_embedding_bank.pth}"
GPU_IDS="${GPU_IDS:-1,2,3,4}"
DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/mnt/data6/yc/datasets/DETECTRON2_DATASETS}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
RESUME="${RESUME:-0}"

for required_file in "${DECLIP_WEIGHTS}" "${ATTR_DATABASE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file does not exist: ${required_file}" >&2
    exit 2
  fi
done

exec env \
  GPU_IDS="${GPU_IDS}" \
  DETECTRON2_DATASETS="${DETECTRON2_DATASETS}" \
  OPENCLIP_PRETRAINED="${DECLIP_WEIGHTS}" \
  ATTR_DATABASE="${ATTR_DATABASE}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  RESUME="${RESUME}" \
  bash "${PROJECT_ROOT}/scripts/train_vitb_attr_fusion_4gpu_bs1.sh" "$@"

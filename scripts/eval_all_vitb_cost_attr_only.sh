#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [NUM_GPUS] [OUTPUT_ROOT] [EXTRA_CFG_OPTS ...]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CHECKPOINT="$1"
NUM_GPUS="${2:-1}"
OUTPUT_ROOT="${3:-${PROJECT_ROOT}/output/eval_all_vitb_cost_attr_only}"
shift $(( $# >= 3 ? 3 : $# ))

export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/media/SD6T/yc/datasets/DETECTRON2_DATASETS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/eva_vitb_384_cost_attr_only.yaml}"
ATTR_DATABASE="${ATTR_DATABASE:-${PROJECT_ROOT}/datasets/attr_database_eva02_clip_b16.pt}"
OPENCLIP_PRETRAINED="${OPENCLIP_PRETRAINED:-${PROJECT_ROOT}/pretrained/EVA02_CLIP_B_psz16_s8B.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUTPUT_ROOT}"

evaluate() {
  local label="$1"
  local class_json="$2"
  local dataset_name="$3"
  local output_dir="${OUTPUT_ROOT}/${label}"
  shift 3

  mkdir -p "${output_dir}"
  echo "===== Evaluating ${label}: ${dataset_name} ====="
  "${PYTHON_BIN}" train_net.py \
    --config-file "${CONFIG}" \
    --num-gpus "${NUM_GPUS}" \
    --dist-url auto \
    --eval-only \
    OUTPUT_DIR "${output_dir}" \
    MODEL.WEIGHTS "${CHECKPOINT}" \
    MODEL.SEM_SEG_HEAD.CACHE_DIR "${OPENCLIP_PRETRAINED}" \
    MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTR_DATABASE}" \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "${class_json}" \
    DATASETS.TEST "(\"${dataset_name}\",)" \
    TEST.SLIDING_WINDOW True \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
    "$@"
}

evaluate coco_stuff_171 datasets/coco.json coco_2017_test_stuff_all_sem_seg "$@"
evaluate ade20k_150 datasets/ade150.json ade20k_150_test_sem_seg "$@"
evaluate ade20k_847 datasets/ade847.json ade20k_full_sem_seg_freq_val_all "$@"
evaluate pascal_voc_20 datasets/voc20.json voc_2012_test_sem_seg "$@"
evaluate pascal_voc_20_background datasets/voc20b.json voc_2012_test_background_sem_seg "$@"
evaluate pascal_context_59 datasets/pc59.json context_59_test_sem_seg "$@"
evaluate pascal_context_459 datasets/pc459.json context_459_test_sem_seg "$@"

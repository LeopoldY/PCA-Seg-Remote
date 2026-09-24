#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -z "${PYTHON_BIN:-}" && -x /opt/miniconda3/envs/yc_d2/bin/python ]]; then
  PYTHON_BIN=/opt/miniconda3/envs/yc_d2/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

CONFIG="${CONFIG:-configs/eva_vitb_384_bs4_scaled.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/output/base_4gpu_bs4_scaled/model_final.pth}"
ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/excel/ms_coco_desc_eva02_clip_b16_gpt4.0_cluster_224_embedding_bank.pth}"
OPENCLIP_PRETRAINED="${OPENCLIP_PRETRAINED:-${PROJECT_ROOT}/pretrained/EVA02_CLIP_B_psz16_s8B.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/base_4gpu_bs4_scaled/eval_all_excel_coco_attr}"
NUM_GPUS="${NUM_GPUS:-4}"

mkdir -p "${OUTPUT_ROOT}"
SUMMARY="${OUTPUT_ROOT}/summary.txt"
: > "${SUMMARY}"
failures=()

evaluate() {
  local label="$1"
  local class_json="$2"
  local dataset_name="$3"
  local sliding_window="$4"
  local pooling_sizes="$5"
  local output_dir="${OUTPUT_ROOT}/${label}"

  mkdir -p "${output_dir}"
  printf '\n===== %s (%s) =====\n' "${label}" "${dataset_name}" | tee -a "${SUMMARY}"

  if "${PYTHON_BIN}" train_net.py \
    --config-file "${CONFIG}" \
    --num-gpus "${NUM_GPUS}" \
    --dist-url auto \
    --eval-only \
    OUTPUT_DIR "${output_dir}" \
    MODEL.WEIGHTS "${CHECKPOINT}" \
    MODEL.SEM_SEG_HEAD.CACHE_DIR "${OPENCLIP_PRETRAINED}" \
    MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED True \
    MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTRIBUTE_DATABASE}" \
    MODEL.SEM_SEG_HEAD.ATTR_FUSION.NUM_CLUSTERS 224 \
    MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "${class_json}" \
    DATASETS.TEST "(\"${dataset_name}\",)" \
    TEST.SLIDING_WINDOW "${sliding_window}" \
    MODEL.SEM_SEG_HEAD.POOLING_SIZES "${pooling_sizes}" \
    2>&1 | tee "${output_dir}/console.log"; then
    grep 'copypaste:' "${output_dir}/log.txt" | tail -n 3 | tee -a "${SUMMARY}" || true
  else
    failures+=("${label}")
    printf 'FAILED: %s\n' "${label}" | tee -a "${SUMMARY}"
  fi
}

# The same official ExCEL COCO database is used for every dataset. No
# ADE/Context-specific descriptors are generated or adapted.
evaluate coco_stuff datasets/coco.json coco_2017_test_stuff_all_sem_seg False '[2,2]'
evaluate ade20k_150 datasets/ade150.json ade20k_150_test_sem_seg True '[1,1]'
evaluate ade20k_847 datasets/ade847.json ade20k_full_sem_seg_freq_val_all True '[1,1]'
evaluate pascal_voc20 datasets/voc20.json voc_2012_test_sem_seg True '[1,1]'
evaluate pascal_voc20_background datasets/voc20b.json voc_2012_test_background_sem_seg True '[1,1]'
evaluate pascal_context_59 datasets/pc59.json context_59_test_sem_seg True '[1,1]'
evaluate pascal_context_459 datasets/pc459.json context_459_test_sem_seg True '[1,1]'

if ((${#failures[@]})); then
  printf '\nFailed datasets: %s\n' "${failures[*]}" | tee -a "${SUMMARY}"
  exit 1
fi

printf '\nAll ExCEL attribute-fusion evaluations completed successfully.\n' | tee -a "${SUMMARY}"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
GPU_IDS="${GPU_IDS:-1,2,3,4}"
DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/mnt/data6/yc/datasets/DETECTRON2_DATASETS}"
CONFIG="${CONFIG:-configs/eva_vitb_384_bs4_scaled.yaml}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/declip_eva_b16_dinov2b_coco_full_retry1_excel_tse_4gpu_bs1_scaled}"
CHECKPOINT="${CHECKPOINT:-${TRAIN_OUTPUT}/model_final.pth}"
ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${TRAIN_OUTPUT}/ms_coco_desc_declip_eva_b16_cluster_224_embedding_bank.pth}"
DECLIP_PRETRAINED="${DECLIP_PRETRAINED:-${TRAIN_OUTPUT}/declip_eva_b16_epoch6_state_dict.pt}"
COCO_TRAIN_LOG="${COCO_TRAIN_LOG:-${TRAIN_OUTPUT}/log.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_OUTPUT}/eval_all}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU ids: ${GPU_IDS}" >&2
  exit 2
fi

for required_file in \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  "${ATTRIBUTE_DATABASE}" \
  "${DECLIP_PRETRAINED}" \
  "${COCO_TRAIN_LOG}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file does not exist: ${required_file}" >&2
    exit 2
  fi
done
if [[ ! -d "${DETECTRON2_DATASETS}" ]]; then
  echo "Dataset root does not exist: ${DETECTRON2_DATASETS}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
SUMMARY="${OUTPUT_ROOT}/summary.txt"
: > "${SUMMARY}"
failures=()

# The final model was already evaluated on COCO-Stuff at iteration 159999.
printf '===== coco_stuff (coco_2017_test_stuff_all_sem_seg) =====\n' | tee -a "${SUMMARY}"
grep 'copypaste:' "${COCO_TRAIN_LOG}" | tail -n 3 | tee -a "${SUMMARY}"

evaluate() {
  local label="$1"
  local class_json="$2"
  local dataset_name="$3"
  local sliding_window="$4"
  local pooling_sizes="$5"
  local output_dir="${OUTPUT_ROOT}/${label}"

  mkdir -p "${output_dir}"
  printf '\n===== %s (%s) =====\n' "${label}" "${dataset_name}" | tee -a "${SUMMARY}"

  if DETECTRON2_DATASETS="${DETECTRON2_DATASETS}" \
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    "${PYTHON_BIN}" train_net.py \
      --config-file "${CONFIG}" \
      --num-gpus 4 \
      --dist-url auto \
      --eval-only \
      OUTPUT_DIR "${output_dir}" \
      MODEL.WEIGHTS "${CHECKPOINT}" \
      MODEL.SEM_SEG_HEAD.CACHE_DIR "${DECLIP_PRETRAINED}" \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED True \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTRIBUTE_DATABASE}" \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.NUM_CLUSTERS 224 \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.TOP_K 0.9 \
      MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "${class_json}" \
      DATASETS.TEST "(\"${dataset_name}\",)" \
      TEST.SLIDING_WINDOW "${sliding_window}" \
      MODEL.SEM_SEG_HEAD.POOLING_SIZES "${pooling_sizes}" \
      2>&1 | tee "${output_dir}/console.log"; then
    grep 'copypaste:' "${output_dir}/log.txt" | tail -n 3 | tee -a "${SUMMARY}"
  else
    failures+=("${label}")
    printf 'FAILED: %s\n' "${label}" | tee -a "${SUMMARY}"
  fi
}

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

printf '\nAll DeCLIP PCA-Seg attribute-fusion evaluations completed successfully.\n' | tee -a "${SUMMARY}"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
GPU_IDS="${GPU_IDS:-1,2,3,4}"
DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/mnt/data6/yc/datasets/DETECTRON2_DATASETS}"
CONFIG="${CONFIG:-configs/eva_vitb_384_bs4_scaled.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/output/eva_vitb_384_excel_tse_4gpu_bs1_scaled/model_final.pth}"
EVA_CHECKPOINT="${EVA_CHECKPOINT:-${PROJECT_ROOT}/pretrained/EVA02_CLIP_B_psz16_s8B.pt}"
DESCRIPTORS="${DESCRIPTORS:-${PROJECT_ROOT}/attributes_text/excel/descriptors_pascal_voc_gpt4.0_cluster_a_photo_of4.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/eva_vitb_384_excel_tse_4gpu_bs1_scaled/eval_voc_specific_excel}"
ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${OUTPUT_ROOT}/pascal_voc_desc_eva02_clip_b16_current_model_cluster_112_embedding_bank.pth}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU ids: ${GPU_IDS}" >&2
  exit 2
fi

for required_file in \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  "${EVA_CHECKPOINT}" \
  "${DESCRIPTORS}"; do
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
PROVENANCE="${OUTPUT_ROOT}/attribute_database_provenance.txt"
: > "${SUMMARY}"

CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" \
  "${PYTHON_BIN}" tools/build_attribute_database.py \
    --descriptors-json "${DESCRIPTORS}" \
    --output "${ATTRIBUTE_DATABASE}" \
    --model-name EVA02-CLIP-B-16 \
    --checkpoint "${EVA_CHECKPOINT}" \
    --model-checkpoint "${CHECKPOINT}" \
    --backend eva \
    --device cuda \
    --num-clusters 112 \
    2>&1 | tee "${OUTPUT_ROOT}/build_attribute_database.log"

{
  echo "descriptors=${DESCRIPTORS}"
  echo "base_eva_checkpoint=${EVA_CHECKPOINT}"
  echo "pca_seg_checkpoint=${CHECKPOINT}"
  echo "attribute_database=${ATTRIBUTE_DATABASE}"
  sha256sum "${DESCRIPTORS}" "${CHECKPOINT}" "${ATTRIBUTE_DATABASE}"
  "${PYTHON_BIN}" -c "import torch; x=torch.load('${ATTRIBUTE_DATABASE}', map_location='cpu', weights_only=True); print('cluster_bank_shape=' + str(tuple(x[0].shape))); print('class_flags_shape=' + str(tuple(x[1].shape)))"
} | tee "${PROVENANCE}"

failures=()
evaluate() {
  local label="$1"
  local class_json="$2"
  local dataset_name="$3"
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
      MODEL.SEM_SEG_HEAD.CACHE_DIR "${EVA_CHECKPOINT}" \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED True \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTRIBUTE_DATABASE}" \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.NUM_CLUSTERS 112 \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.TOP_K 0.9 \
      MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "${class_json}" \
      DATASETS.TEST "(\"${dataset_name}\",)" \
      TEST.SLIDING_WINDOW True \
      MODEL.SEM_SEG_HEAD.POOLING_SIZES '[1,1]' \
      2>&1 | tee "${output_dir}/console.log"; then
    grep 'copypaste:' "${output_dir}/log.txt" | tail -n 3 | tee -a "${SUMMARY}"
  else
    failures+=("${label}")
    printf 'FAILED: %s\n' "${label}" | tee -a "${SUMMARY}"
  fi
}

evaluate pascal_voc20 datasets/voc20.json voc_2012_test_sem_seg
evaluate pascal_voc20_background datasets/voc20b.json voc_2012_test_background_sem_seg

if ((${#failures[@]})); then
  printf '\nFailed datasets: %s\n' "${failures[*]}" | tee -a "${SUMMARY}"
  exit 1
fi

printf '\nVOC-specific ExCEL attribute evaluations completed successfully.\n' | tee -a "${SUMMARY}"

#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID}" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
case "$1" in
  DLRSD)
    TAG=dlrsd
    DEFAULT_GPU=0
    TRAIN_OUTPUT="${PROJECT_ROOT}/output/clip_vitb_384_dlrsd_attr20_dual_teacher_dual_feature_moe_content1_context0p1_2gpu_bs4_20260910_223855"
    WEIGHT=model_final.pth
    ;;
  iSAID)
    TAG=isaid
    DEFAULT_GPU=1
    TRAIN_OUTPUT="${PROJECT_ROOT}/output/clip_vitb_384_isaid_attr20_dual_teacher_dual_feature_moe_content1_context0p1_2gpu_bs4_20260910_225555"
    WEIGHT=model_0024999.pth
    ;;
esac
GPU_ID="${GPU_ID:-${DEFAULT_GPU}}"
[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be one GPU index" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
CONFIG="configs/clip_vitb_384_${TAG}_attr20_dual_teacher_dual_feature_moe.yaml"
CHECKPOINT="${TRAIN_OUTPUT}/${WEIGHT}"
ATTRIBUTE_DATABASE="${PROJECT_ROOT}/attributes_text/rskt_seg/${1}_train_desc_remoteclip_b32_cluster_20_embedding_bank.pth"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/data6/yc/open-vocab/RSKT-Seg/pretrained/ViT-B-16.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_OUTPUT}/eval_best_all8}"
DATASETS=(DLRSD iSAID Potsdam Vaihingen UDD5 LoveDA uavid VDD)
for path in "${PYTHON_BIN}" "${CONFIG}" "${CHECKPOINT}" "${ATTRIBUTE_DATABASE}" "${CLIP_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "Missing file: ${path}" >&2; exit 2; }
done
[[ -d "${DATASET_ROOT}" ]] || { echo "Missing dataset root: ${DATASET_ROOT}" >&2; exit 2; }
for dataset in "${DATASETS[@]}"; do
  [[ -f "datasets/${dataset}.json" ]] || { echo "Missing class JSON: ${dataset}" >&2; exit 2; }
done
echo "Training dataset: $1; GPU: ${GPU_ID}; checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_ROOT}; datasets: ${DATASETS[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN: path checks passed; evaluation not started."
  exit 0
fi
mkdir -p "${OUTPUT_ROOT}"
SUMMARY="${OUTPUT_ROOT}/summary.txt"
failures=()
printf 'Checkpoint: %s\nGPU: %s\n' "${CHECKPOINT}" "${GPU_ID}" > "${SUMMARY}"
for dataset in "${DATASETS[@]}"; do
  output_dir="${OUTPUT_ROOT}/${dataset}"
  mkdir -p "${output_dir}"
  if CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    OVSISBENCH_DATASETS="${DATASET_ROOT}" DETECTRON2_DATASETS="${DATASET_ROOT}" \
    "${PYTHON_BIN}" train_net.py \
      --config-file "${CONFIG}" --num-gpus 1 --eval-only \
      MODEL.WEIGHTS "${CHECKPOINT}" OUTPUT_DIR "${output_dir}" \
      MODEL.SEM_SEG_HEAD.CACHE_DIR "${CLIP_CHECKPOINT}" \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED True \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTRIBUTE_DATABASE}" \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.NUM_CLUSTERS 20 \
      MODEL.SEM_SEG_HEAD.FEATURE_FUSION.TYPE dual_feature_moe \
      MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/${dataset}.json" \
      DATASETS.TEST "(\"${dataset}_all_sem_seg\",)" \
      MODEL.SEM_SEG_HEAD.RS_DINO.ENABLED False \
      MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.ENABLED False \
      MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.ENABLED False \
      TEST.AUG.ENABLED False TEST.SLIDING_WINDOW False \
      MODEL.SEM_SEG_HEAD.POOLING_SIZES '[1,1]' \
      2>&1 | tee "${output_dir}/console.log"; then
    printf '\nPASS: %s\n' "${dataset}" | tee -a "${SUMMARY}"
    grep 'copypaste:' "${output_dir}/console.log" | tail -n 3 >> "${SUMMARY}" || true
  else
    failures+=("${dataset}")
    printf '\nFAILED: %s\n' "${dataset}" | tee -a "${SUMMARY}"
  fi
done
if ((${#failures[@]})); then
  echo "Failed datasets: ${failures[*]}" | tee -a "${SUMMARY}"
  exit 1
fi
echo "All eight datasets completed." | tee -a "${SUMMARY}"

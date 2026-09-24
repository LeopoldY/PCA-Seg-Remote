#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
GPU_IDS="${GPU_IDS:-1,2,3,4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
CONFIG="${CONFIG:?Set CONFIG to the training config path}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to model_final.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT for evaluation results}"
ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:?Set ATTRIBUTE_DATABASE to the dataset-specific 64-center bank}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"
NUM_CLUSTERS="${NUM_CLUSTERS:-64}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU ids: ${GPU_IDS}" >&2
  exit 2
fi
if [[ "${NUM_CLUSTERS}" -ne 64 ]]; then
  echo "This evaluation setup requires NUM_CLUSTERS=64, got ${NUM_CLUSTERS}" >&2
  exit 2
fi

for path in \
  "${PYTHON_BIN}" \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  "${ATTRIBUTE_DATABASE}" \
  "${DECLIP_CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}"
SUMMARY="${OUTPUT_ROOT}/summary.txt"
: > "${SUMMARY}"
failures=()

evaluate() {
  local label="$1"
  local class_json="$2"
  local dataset_name="$3"
  local output_dir="${OUTPUT_ROOT}/${label}"

  mkdir -p "${output_dir}"
  printf '\n===== %s (%s) =====\n' "${label}" "${dataset_name}" | tee -a "${SUMMARY}"

  if OVSISBENCH_DATASETS="${DATASET_ROOT}" \
    DETECTRON2_DATASETS="${DATASET_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    "${PYTHON_BIN}" train_net.py \
      --config-file "${CONFIG}" \
      --num-gpus 4 \
      --dist-url auto \
      --eval-only \
      OUTPUT_DIR "${output_dir}" \
      MODEL.WEIGHTS "${CHECKPOINT}" \
      MODEL.SEM_SEG_HEAD.CACHE_DIR "${DECLIP_CHECKPOINT}" \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED True \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTRIBUTE_DATABASE}" \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.NUM_CLUSTERS "${NUM_CLUSTERS}" \
      MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "${class_json}" \
      DATASETS.TEST "(\"${dataset_name}\",)" \
      TEST.SLIDING_WINDOW False \
      MODEL.SEM_SEG_HEAD.POOLING_SIZES '[1,1]' \
      2>&1 | tee "${output_dir}/console.log"; then
    grep 'copypaste:' "${output_dir}/log.txt" | tail -n 3 | tee -a "${SUMMARY}" || true
  else
    failures+=("${label}")
    printf 'FAILED: %s\n' "${label}" | tee -a "${SUMMARY}"
  fi
}

evaluate DLRSD datasets/DLRSD.json DLRSD_all_sem_seg
evaluate iSAID datasets/iSAID.json iSAID_all_sem_seg
evaluate Potsdam datasets/Potsdam.json Potsdam_all_sem_seg
evaluate Vaihingen datasets/Vaihingen.json Vaihingen_all_sem_seg
evaluate UDD5 datasets/UDD5.json UDD5_all_sem_seg
evaluate LoveDA datasets/LoveDA.json LoveDA_all_sem_seg
evaluate UAVid datasets/uavid.json uavid_all_sem_seg
evaluate VDD datasets/VDD.json VDD_all_sem_seg

if ((${#failures[@]})); then
  printf '\nFailed datasets: %s\n' "${failures[*]}" | tee -a "${SUMMARY}"
  exit 1
fi

printf '\nAll eight OVSISBench evaluations completed successfully.\n' | tee -a "${SUMMARY}"

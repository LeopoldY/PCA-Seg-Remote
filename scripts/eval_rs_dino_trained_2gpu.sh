#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID}" >&2
  exit 2
fi

DATASET="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/evaluation_results/rs_dino_2gpu_bs4}"

case "${DATASET}" in
  DLRSD)
    GPU_IDS="${GPU_IDS:-0,1}"
    CONFIG="${CONFIG:-configs/eva_vitb_384_dlrsd_attr64_rs_dino_rskt_protocol.yaml}"
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr64_rs_dino_rskt_protocol_2gpu_bs4}"
    ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/DLRSD_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"
    EVAL_OUTPUT="${EVAL_OUTPUT:-${TRAIN_OUTPUT}/evaluation_ovsisbench_all_2gpu}"
    RESULT_FILE="${RESULT_FILE:-${RESULT_ROOT}/DLRSD_checkpoint_all_8_datasets.txt}"
    ;;
  iSAID)
    GPU_IDS="${GPU_IDS:-2,3}"
    CONFIG="${CONFIG:-configs/eva_vitb_384_isaid_attr64_rs_dino_rskt_protocol.yaml}"
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr64_rs_dino_rskt_protocol_2gpu_bs4}"
    ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/iSAID_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"
    EVAL_OUTPUT="${EVAL_OUTPUT:-${TRAIN_OUTPUT}/evaluation_ovsisbench_all_2gpu}"
    RESULT_FILE="${RESULT_FILE:-${RESULT_ROOT}/iSAID_checkpoint_all_8_datasets.txt}"
    ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 2 ]]; then
  echo "GPU_IDS must contain exactly two comma-separated GPU ids: ${GPU_IDS}" >&2
  exit 2
fi

if [[ -z "${CHECKPOINT:-}" ]]; then
  if [[ ! -f "${TRAIN_OUTPUT}/last_checkpoint" ]]; then
    echo "Missing last_checkpoint: ${TRAIN_OUTPUT}/last_checkpoint" >&2
    exit 2
  fi
  CHECKPOINT_NAME="$(< "${TRAIN_OUTPUT}/last_checkpoint")"
  CHECKPOINT="${TRAIN_OUTPUT}/${CHECKPOINT_NAME}"
fi

for path in \
  "${PYTHON_BIN}" \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  "${ATTRIBUTE_DATABASE}" \
  "${DECLIP_CHECKPOINT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file does not exist: ${path}" >&2
    exit 2
  fi
done

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 2
fi

mkdir -p "${EVAL_OUTPUT}" "${RESULT_ROOT}" "$(dirname "${RESULT_FILE}")"

{
  echo "===== ${DATASET}-trained checkpoint: all 8 OVSISBench datasets ====="
  echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "GPUs: ${GPU_IDS}"
  echo "Checkpoint: ${CHECKPOINT}"
  echo "Config: ${CONFIG}"
  echo "Attribute database: ${ATTRIBUTE_DATABASE}"
  echo "Evaluation output: ${EVAL_OUTPUT}"
  echo
} | tee "${RESULT_FILE}"

failures=()

evaluate() {
  local label="$1"
  local class_json="$2"
  local dataset_name="$3"
  local output_dir="${EVAL_OUTPUT}/${label}"

  mkdir -p "${output_dir}"
  printf '\n===== %s (%s) =====\n' \
    "${label}" "${dataset_name}" | tee -a "${RESULT_FILE}"

  if OVSISBENCH_DATASETS="${DATASET_ROOT}" \
    DETECTRON2_DATASETS="${DATASET_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    "${PYTHON_BIN}" train_net.py \
      --config-file "${CONFIG}" \
      --num-gpus 2 \
      --dist-url auto \
      --eval-only \
      OUTPUT_DIR "${output_dir}" \
      MODEL.WEIGHTS "${CHECKPOINT}" \
      MODEL.SEM_SEG_HEAD.CACHE_DIR "${DECLIP_CHECKPOINT}" \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED True \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTRIBUTE_DATABASE}" \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.NUM_CLUSTERS 64 \
      MODEL.SEM_SEG_HEAD.RS_DINO.ENABLED False \
      MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.ENABLED False \
      MODEL.SEM_SEG_HEAD.CLIP_ROTATION.ENABLED True \
      MODEL.SEM_SEG_HEAD.FEATURE_FUSION.TYPE dual_feature_moe \
      MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "${class_json}" \
      DATASETS.TEST "(\"${dataset_name}\",)" \
      TEST.SLIDING_WINDOW False \
      MODEL.SEM_SEG_HEAD.POOLING_SIZES '[1,1]' \
      2>&1 | tee "${output_dir}/console.log"; then
    grep -E 'Evaluation results for|copypaste:' \
      "${output_dir}/log.txt" | tail -n 4 | tee -a "${RESULT_FILE}" || true
  else
    failures+=("${label}")
    printf 'FAILED: %s; see %s/console.log\n' \
      "${label}" "${output_dir}" | tee -a "${RESULT_FILE}"
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

{
  echo
  echo "===== all-8 evaluation status ====="
  echo "Finished: $(date '+%Y-%m-%d %H:%M:%S %z')"
  if ((${#failures[@]})); then
    echo "Failed datasets: ${failures[*]}"
  else
    echo "All eight OVSISBench evaluations completed successfully."
  fi
} | tee -a "${RESULT_FILE}"

if ((${#failures[@]})); then
  exit 1
fi

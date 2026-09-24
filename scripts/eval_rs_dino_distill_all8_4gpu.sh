#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID} [detectron2 config overrides ...]" >&2
  exit 2
fi

TRAIN_DATASET="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-1,2,3,4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"

case "${TRAIN_DATASET}" in
  DLRSD)
    CONFIG="${CONFIG:-configs/eva_vitb_384_dlrsd_attr64_rs_dino_rskt_protocol.yaml}"
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-/mnt/data6/yc/open-vocab/PCA-Seg-Remote/output/eva_vitb_384_dlrsd_attr64_rs_dino_rskt_protocol_4gpu_bs2}"
    ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/DLRSD_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"
    ;;
  iSAID)
    CONFIG="${CONFIG:-configs/eva_vitb_384_isaid_attr64_rs_dino_rskt_protocol.yaml}"
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-/mnt/data6/yc/open-vocab/PCA-Seg-Remote/output/eva_vitb_384_isaid_attr64_rs_dino_rskt_protocol_4gpu_bs2}"
    ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/iSAID_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"
    ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_OUTPUT}/eval_ovsisbench_all8}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
if ((NUM_GPUS < 1)); then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 2
fi

resolve_checkpoint() {
  if [[ -n "${CHECKPOINT:-}" ]]; then
    printf '%s\n' "${CHECKPOINT}"
    return
  fi
  if [[ -f "${TRAIN_OUTPUT}/model_final.pth" ]]; then
    printf '%s\n' "${TRAIN_OUTPUT}/model_final.pth"
    return
  fi
  if [[ -f "${TRAIN_OUTPUT}/last_checkpoint" ]]; then
    local checkpoint_name
    checkpoint_name="$(< "${TRAIN_OUTPUT}/last_checkpoint")"
    if [[ "${checkpoint_name}" = /* ]]; then
      printf '%s\n' "${checkpoint_name}"
    else
      printf '%s\n' "${TRAIN_OUTPUT}/${checkpoint_name}"
    fi
    return
  fi
  echo "Cannot find model_final.pth or last_checkpoint in ${TRAIN_OUTPUT}" >&2
  return 1
}

CHECKPOINT="$(resolve_checkpoint)"

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

mkdir -p "${OUTPUT_ROOT}"
SUMMARY="${OUTPUT_ROOT}/summary.txt"
: > "${SUMMARY}"
failures=()

{
  echo "===== ${TRAIN_DATASET}-trained RS-DINO-distilled model: OVSISBench all-8 evaluation ====="
  echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "GPUs: ${GPU_IDS} (${NUM_GPUS} processes)"
  echo "Checkpoint: ${CHECKPOINT}"
  echo "Config: ${CONFIG}"
  echo "Attribute database: ${ATTRIBUTE_DATABASE}"
  echo "Dataset root: ${DATASET_ROOT}"
  echo "Output root: ${OUTPUT_ROOT}"
  echo "Inference: student only; RS-DINO teacher disabled"
} | tee -a "${SUMMARY}"

evaluate() {
  local label="$1"
  local class_json="$2"
  local dataset_name="$3"
  shift 3
  local output_dir="${OUTPUT_ROOT}/${label}"

  mkdir -p "${output_dir}"
  printf '\n===== %s (%s) =====\n' "${label}" "${dataset_name}" | tee -a "${SUMMARY}"

  if OVSISBENCH_DATASETS="${DATASET_ROOT}" \
    DETECTRON2_DATASETS="${DATASET_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    "${PYTHON_BIN}" train_net.py \
      --config-file "${CONFIG}" \
      --num-gpus "${NUM_GPUS}" \
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
      MODEL.SEM_SEG_HEAD.CLIP_ROTATION.NUM_DIRECTIONS 4 \
      MODEL.SEM_SEG_HEAD.FEATURE_FUSION.TYPE dual_feature_moe \
      MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "${class_json}" \
      DATASETS.TEST "(\"${dataset_name}\",)" \
      TEST.SLIDING_WINDOW False \
      MODEL.SEM_SEG_HEAD.POOLING_SIZES '[1,1]' \
      "$@" \
      2>&1 | tee "${output_dir}/console.log"; then
    grep -E 'Evaluation results for|copypaste:' \
      "${output_dir}/log.txt" | tail -n 4 | tee -a "${SUMMARY}" || true
  else
    failures+=("${label}")
    printf 'FAILED: %s; see %s/console.log\n' \
      "${label}" "${output_dir}" | tee -a "${SUMMARY}"
  fi
}

evaluate DLRSD datasets/DLRSD.json DLRSD_all_sem_seg "$@"
evaluate iSAID datasets/iSAID.json iSAID_all_sem_seg "$@"
evaluate Potsdam datasets/Potsdam.json Potsdam_all_sem_seg "$@"
evaluate Vaihingen datasets/Vaihingen.json Vaihingen_all_sem_seg "$@"
evaluate UDD5 datasets/UDD5.json UDD5_all_sem_seg "$@"
evaluate LoveDA datasets/LoveDA.json LoveDA_all_sem_seg "$@"
evaluate UAVid datasets/uavid.json uavid_all_sem_seg "$@"
evaluate VDD datasets/VDD.json VDD_all_sem_seg "$@"

{
  echo
  echo "===== all-8 evaluation status ====="
  echo "Finished: $(date '+%Y-%m-%d %H:%M:%S %z')"
  if ((${#failures[@]})); then
    echo "Failed datasets: ${failures[*]}"
  else
    echo "All eight OVSISBench evaluations completed successfully."
  fi
} | tee -a "${SUMMARY}"

if ((${#failures[@]})); then
  exit 1
fi

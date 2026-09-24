#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-1,2,3,4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"

DLRSD_CONFIG="${DLRSD_CONFIG:-configs/eva_vitb_384_dlrsd_attr64_declip_rskt_protocol.yaml}"
DLRSD_CHECKPOINT="${DLRSD_CHECKPOINT:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr64_declip_rskt_protocol_4gpu_bs2/model_final.pth}"
DLRSD_ATTRIBUTE_DATABASE="${DLRSD_ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/DLRSD_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"
DLRSD_OUTPUT="${DLRSD_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr64_declip_rskt_protocol_4gpu_bs2/eval_ovsisbench_all_attr64_declip}"

ISAID_CONFIG="${ISAID_CONFIG:-configs/eva_vitb_384_isaid_attr64_declip_rskt_protocol.yaml}"
ISAID_CHECKPOINT="${ISAID_CHECKPOINT:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr64_declip_rskt_protocol_4gpu_bs2/model_final.pth}"
ISAID_ATTRIBUTE_DATABASE="${ISAID_ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/iSAID_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"
ISAID_OUTPUT="${ISAID_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr64_declip_rskt_protocol_4gpu_bs2/eval_ovsisbench_all_attr64_declip}"

for path in \
  "${PYTHON_BIN}" \
  "${DECLIP_CHECKPOINT}" \
  "${DLRSD_CONFIG}" \
  "${DLRSD_CHECKPOINT}" \
  "${DLRSD_ATTRIBUTE_DATABASE}" \
  "${ISAID_CONFIG}" \
  "${ISAID_CHECKPOINT}" \
  "${ISAID_ATTRIBUTE_DATABASE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file does not exist: ${path}" >&2
    exit 2
  fi
done

run_evaluation() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local attribute_database="$4"
  local output_root="$5"

  echo "===== Evaluating ${label} checkpoint on all OVSISBench datasets ====="
  env \
    GPU_IDS="${GPU_IDS}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT}" \
    NUM_CLUSTERS=64 \
    CONFIG="${config}" \
    CHECKPOINT="${checkpoint}" \
    ATTRIBUTE_DATABASE="${attribute_database}" \
    OUTPUT_ROOT="${output_root}" \
    bash scripts/eval_ovsisbench_all_attr64_declip_4gpu.sh
}

# Run sequentially because both evaluations use the same four GPUs.
run_evaluation \
  DLRSD \
  "${DLRSD_CONFIG}" \
  "${DLRSD_CHECKPOINT}" \
  "${DLRSD_ATTRIBUTE_DATABASE}" \
  "${DLRSD_OUTPUT}"

run_evaluation \
  iSAID \
  "${ISAID_CONFIG}" \
  "${ISAID_CHECKPOINT}" \
  "${ISAID_ATTRIBUTE_DATABASE}" \
  "${ISAID_OUTPUT}"

echo "===== Both checkpoints completed ====="
echo "DLRSD summary: ${DLRSD_OUTPUT}/summary.txt"
echo "iSAID summary: ${ISAID_OUTPUT}/summary.txt"

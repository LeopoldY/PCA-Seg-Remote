#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-1,2,3,4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"

DLRSD_CONFIG="${DLRSD_CONFIG:-configs/eva_vitb_384_dlrsd_attr112_rskt_protocol.yaml}"
DLRSD_CHECKPOINT="${DLRSD_CHECKPOINT:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr112_rskt_protocol_4gpu_bs2/model_final.pth}"
DLRSD_OUTPUT="${DLRSD_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr112_rskt_protocol_4gpu_bs2/eval_ovsisbench_all}"

ISAID_CONFIG="${ISAID_CONFIG:-configs/eva_vitb_384_isaid_attr112_rskt_protocol.yaml}"
ISAID_CHECKPOINT="${ISAID_CHECKPOINT:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr112_rskt_protocol_4gpu_bs2/model_final.pth}"
ISAID_OUTPUT="${ISAID_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr112_rskt_protocol_4gpu_bs2/eval_ovsisbench_all}"

for path in \
  "${DLRSD_CONFIG}" \
  "${DLRSD_CHECKPOINT}" \
  "${ISAID_CONFIG}" \
  "${ISAID_CHECKPOINT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file does not exist: ${path}" >&2
    exit 2
  fi
done

run_evaluation() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local output_root="$4"

  echo "===== Evaluating ${label} checkpoint on all OVSISBench datasets ====="
  env \
    GPU_IDS="${GPU_IDS}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    CONFIG="${config}" \
    CHECKPOINT="${checkpoint}" \
    OUTPUT_ROOT="${output_root}" \
    bash scripts/eval_ovsisbench_all_attr112_4gpu.sh
}

# Run sequentially because both evaluations use the same four GPUs.
run_evaluation DLRSD "${DLRSD_CONFIG}" "${DLRSD_CHECKPOINT}" "${DLRSD_OUTPUT}"
run_evaluation iSAID "${ISAID_CONFIG}" "${ISAID_CHECKPOINT}" "${ISAID_OUTPUT}"

echo "===== Both checkpoints completed ====="
echo "DLRSD summary: ${DLRSD_OUTPUT}/summary.txt"
echo "iSAID summary: ${ISAID_OUTPUT}/summary.txt"

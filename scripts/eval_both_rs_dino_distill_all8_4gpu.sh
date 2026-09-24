#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-1,2,3,4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"

DLRSD_TRAIN_OUTPUT="${DLRSD_TRAIN_OUTPUT:-/mnt/data6/yc/open-vocab/PCA-Seg-Remote/output/eva_vitb_384_dlrsd_attr64_rs_dino_rskt_protocol_4gpu_bs2}"
ISAID_TRAIN_OUTPUT="${ISAID_TRAIN_OUTPUT:-/mnt/data6/yc/open-vocab/PCA-Seg-Remote/output/eva_vitb_384_isaid_attr64_rs_dino_rskt_protocol_4gpu_bs2}"
failures=()

run_evaluation() {
  local dataset="$1"
  local train_output="$2"

  echo "===== Starting ${dataset}-trained checkpoint ====="
  if env \
      GPU_IDS="${GPU_IDS}" \
      DATASET_ROOT="${DATASET_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT}" \
      TRAIN_OUTPUT="${train_output}" \
      OUTPUT_ROOT="${train_output}/eval_ovsisbench_all8" \
      bash scripts/eval_rs_dino_distill_all8_4gpu.sh "${dataset}"; then
    echo "===== ${dataset}-trained checkpoint completed ====="
  else
    failures+=("${dataset}")
    echo "===== ${dataset}-trained checkpoint had evaluation failures =====" >&2
  fi
}

# Both checkpoints are evaluated sequentially because they share the same GPUs.
run_evaluation DLRSD "${DLRSD_TRAIN_OUTPUT}"
run_evaluation iSAID "${ISAID_TRAIN_OUTPUT}"

echo "===== Both checkpoints completed ====="
echo "DLRSD summary: ${DLRSD_TRAIN_OUTPUT}/eval_ovsisbench_all8/summary.txt"
echo "iSAID summary: ${ISAID_TRAIN_OUTPUT}/eval_ovsisbench_all8/summary.txt"

if ((${#failures[@]})); then
  echo "Checkpoints with failures: ${failures[*]}" >&2
  exit 1
fi

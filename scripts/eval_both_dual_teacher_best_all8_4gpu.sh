#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"
failures=()

run_evaluation() {
  local train_dataset="$1"

  echo "===== Starting ${train_dataset}-trained best checkpoint ====="
  if env \
      GPU_IDS="${GPU_IDS}" \
      DATASET_ROOT="${DATASET_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT}" \
      bash scripts/eval_dual_teacher_best_all8_4gpu.sh "${train_dataset}"; then
    echo "===== ${train_dataset}-trained best checkpoint completed ====="
  else
    failures+=("${train_dataset}")
    echo "===== ${train_dataset}-trained evaluation failed =====" >&2
  fi
}

# Evaluate sequentially because both runs use the same four GPUs.
run_evaluation DLRSD
run_evaluation iSAID

echo "===== Both best checkpoints completed ====="
echo "DLRSD summary: output/eva_vitb_384_dlrsd_attr64_dual_teacher_rskt_protocol_4gpu_bs2/eval_best_miou_all8/summary.txt"
echo "iSAID summary: output/eva_vitb_384_isaid_attr64_dual_teacher_rskt_protocol_4gpu_bs2/eval_best_miou_all8/summary.txt"

if ((${#failures[@]})); then
  echo "Failed trained models: ${failures[*]}" >&2
  exit 1
fi

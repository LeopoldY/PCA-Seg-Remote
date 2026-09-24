#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"

DLRSD_TRAIN_OUTPUT="${DLRSD_TRAIN_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr64_dual_teacher_aff_residual_moe_2gpu_bs4}"
ISAID_TRAIN_OUTPUT="${ISAID_TRAIN_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr64_dual_teacher_aff_residual_moe_2gpu_bs4}"
failures=()

run_evaluation() {
  local train_dataset="$1"
  local train_output="$2"

  echo "===== Starting ${train_dataset} AFF-residual-MoE best checkpoint ====="
  if env \
      GPU_IDS="${GPU_IDS}" \
      DATASET_ROOT="${DATASET_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT}" \
      TRAIN_OUTPUT="${train_output}" \
      OUTPUT_ROOT="${train_output}/eval_best_miou_all8" \
      DRY_RUN="${DRY_RUN:-0}" \
      bash scripts/eval_dual_teacher_aff_residual_moe_best_all8_4gpu.sh \
        "${train_dataset}"; then
    echo "===== ${train_dataset} best checkpoint completed ====="
  else
    failures+=("${train_dataset}")
    echo "===== ${train_dataset} evaluation failed =====" >&2
  fi
}

# Both checkpoints are evaluated sequentially on the same four GPUs.
run_evaluation DLRSD "${DLRSD_TRAIN_OUTPUT}"
run_evaluation iSAID "${ISAID_TRAIN_OUTPUT}"

echo "===== Both AFF-residual-MoE best checkpoints completed ====="
echo "DLRSD summary: ${DLRSD_TRAIN_OUTPUT}/eval_best_miou_all8/summary.txt"
echo "iSAID summary: ${ISAID_TRAIN_OUTPUT}/eval_best_miou_all8/summary.txt"

if ((${#failures[@]})); then
  echo "Failed trained models: ${failures[*]}" >&2
  exit 1
fi

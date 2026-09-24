#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID}" >&2
  exit 2
fi

DATASET="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${DATASET}" in
  DLRSD)
    GPU_IDS="${GPU_IDS:-0,1}"
    CONFIG="${CONFIG:-configs/eva_vitb_384_dlrsd_attr64_rs_dino_sparse_transformer.yaml}"
    TRAIN_DATASET='("DLRSD_train_smoke_sem_seg",)'
    TEST_DATASET='("DLRSD_val_smoke_sem_seg",)'
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/smoke_dlrsd_attr64_rs_dino_sparse_transformer_2gpu_bs4}"
    ;;
  iSAID)
    GPU_IDS="${GPU_IDS:-2,3}"
    CONFIG="${CONFIG:-configs/eva_vitb_384_isaid_attr64_rs_dino_sparse_transformer.yaml}"
    TRAIN_DATASET='("iSAID_train_smoke_sem_seg",)'
    TEST_DATASET='("iSAID_val_smoke_sem_seg",)'
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/smoke_isaid_attr64_rs_dino_sparse_transformer_2gpu_bs4}"
    ;;
esac

echo "Smoke dataset: ${DATASET}"
echo "Fusion: dual_route_sparse_transformer"
echo "GPUs: ${GPU_IDS}; batch: 4/GPU, 8 global"
echo "Scope: 8 train images, 4 val images, 2 iterations"
echo "Output: ${OUTPUT_DIR}"

exec env \
  GPU_IDS="${GPU_IDS}" \
  CONFIG="${CONFIG}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  RESUME=0 \
  bash "${PROJECT_ROOT}/scripts/train_rs_dino_rskt_protocol_2gpu_bs4.sh" \
    "${DATASET}" \
    DATASETS.TRAIN "${TRAIN_DATASET}" \
    DATASETS.TEST "${TEST_DATASET}" \
    SOLVER.MAX_ITER 2 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    TEST.EVAL_PERIOD 2 \
    DATALOADER.NUM_WORKERS 2

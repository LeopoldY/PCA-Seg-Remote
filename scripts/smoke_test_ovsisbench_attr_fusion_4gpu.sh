#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/smoke_ovsisbench_rs31_attr112_4gpu_bs2}"

exec env OUTPUT_DIR="${OUTPUT_DIR}" RESUME=0 \
  "${PROJECT_ROOT}/scripts/train_ovsisbench_attr_fusion_4gpu.sh" \
  DATASETS.TRAIN '("ovsisbench_dlrsd_train_smoke_sem_seg","ovsisbench_isaid_train_smoke_sem_seg")' \
  DATASETS.TEST '("ovsisbench_dlrsd_val_smoke_sem_seg","ovsisbench_isaid_val_smoke_sem_seg")' \
  SOLVER.MAX_ITER 2 \
  SOLVER.CHECKPOINT_PERIOD 2 \
  TEST.EVAL_PERIOD 2 \
  DATALOADER.NUM_WORKERS 2

#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:-iSAID}"
if [[ "${DATASET}" != iSAID && "${DATASET}" != DLRSD ]]; then
  echo "Usage: $0 {iSAID|DLRSD}" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU_IDS="${GPU_IDS:-2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export RESUME=0
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/smoke_semantic_kl_${DATASET}_$(date +%Y%m%d_%H%M%S)}"
exec bash "${PROJECT_ROOT}/scripts/train_semantic_kl_2gpu.sh" "${DATASET}" \
  DATASETS.TRAIN "(\"${DATASET}_train_smoke_sem_seg\",)" \
  DATASETS.TEST "(\"${DATASET}_val_smoke_sem_seg\",)" \
  SOLVER.MAX_ITER 2 SOLVER.CHECKPOINT_PERIOD 1000 TEST.EVAL_PERIOD 2 \
  DATALOADER.NUM_WORKERS 2

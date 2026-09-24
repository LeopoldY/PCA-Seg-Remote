#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID}" >&2
  exit 2
fi
DATASET="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${DATASET}" in
  DLRSD) TAG=dlrsd ;;
  iSAID) TAG=isaid ;;
esac
exec env RESUME=0 \
  OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/smoke_rskt_clipl_${TAG}_dual_teacher_aff_residual_moe_2gpu}" \
  bash "${PROJECT_ROOT}/scripts/train_rskt_clipl_dual_teacher_aff_residual_moe_2gpu.sh" \
  "${DATASET}" \
  DATASETS.TRAIN "(\"${DATASET}_train_smoke_sem_seg\",)" \
  DATASETS.TEST "(\"${DATASET}_val_smoke_sem_seg\",)" \
  SOLVER.MAX_ITER 2 SOLVER.CHECKPOINT_PERIOD 1000 TEST.EVAL_PERIOD 2 \
  DATALOADER.NUM_WORKERS 2

#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export GPU_IDS="${GPU_IDS:-2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/output/clip_vitb_openai_attr64_semantic_kl_${RUN_TIMESTAMP}}"
mkdir -p "${RUN_ROOT}"
for DATASET in iSAID DLRSD; do
  echo "[$(date -Is)] Starting ${DATASET} on GPUs ${GPU_IDS}"
  OUTPUT_DIR="${RUN_ROOT}/${DATASET}" RESUME=0 bash \
    scripts/train_semantic_kl_2gpu.sh "${DATASET}" "$@" \
    2>&1 | tee "${RUN_ROOT}/${DATASET}.log"
  echo "[$(date -Is)] Finished ${DATASET}"
done

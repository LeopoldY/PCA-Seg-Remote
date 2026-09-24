#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU_IDS="${GPU_IDS:-2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
exec bash "${PROJECT_ROOT}/scripts/train_semantic_kl_2gpu.sh" iSAID "$@"

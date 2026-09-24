#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export GPU_IDS="${GPU_IDS:-2,3}"
export CONFIG=configs/clip_vitb_384_isaid_attr64_dual_teacher_dual_feature_moe_rs.yaml
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/isaid_clip_vitb_attr64_top0p9_rs8_rot4_$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export RESUME=0
exec bash scripts/train_rskt_clip_dual_teacher_dual_feature_moe_attr64_2gpu.sh iSAID "$@"

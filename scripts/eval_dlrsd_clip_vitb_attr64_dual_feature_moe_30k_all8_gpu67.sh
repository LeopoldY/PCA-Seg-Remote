#!/usr/bin/env bash
# DLRSD-trained CLIP ViT-B/16 + original DualFeatureMoE + RemoteCLIP attr64 + rot4.
# Evaluate the fixed 30,000-step DLRSD checkpoint on all eight datasets.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export GPU_IDS="${GPU_IDS:-6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ ! "${GPU_IDS}" =~ ^[0-9]+,[0-9]+$ || "${#GPUS[@]}" -ne 2 || "${GPUS[0]}" == "${GPUS[1]}" ]]; then
  echo "GPU_IDS must contain two distinct GPU indices" >&2
  exit 2
fi
export TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/clip_vitb_attr64_dual_teacher_dual_feature_moe_rot4_context1e-5_content1e-4_20260912_gpu23/DLRSD}"
export CONFIG="${TRAIN_OUTPUT}/config.yaml"
export CHECKPOINT="${TRAIN_OUTPUT}/model_0029999.pth"
export FEATURE_FUSION_TYPE=dual_feature_moe
export ATTRIBUTE_DATABASE="${PROJECT_ROOT}/attributes_text/rskt_seg/DLRSD_train_desc_remoteclip_b32_cluster_64_embedding_bank.pth"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_OUTPUT}/eval_iter29999_all8_gpu67_$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
for dataset in DLRSD iSAID Potsdam Vaihingen UDD5 LoveDA uavid VDD; do
  [[ -f "datasets/${dataset}.json" ]] || { echo "Missing class JSON: ${dataset}" >&2; exit 2; }
done
exec bash scripts/eval_rskt_clip_checkpoint_all8_2gpu.sh DLRSD TEST.AUG.ENABLED False "$@"

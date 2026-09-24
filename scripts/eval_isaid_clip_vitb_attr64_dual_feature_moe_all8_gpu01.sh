#!/usr/bin/env bash
# iSAID-trained CLIP ViT-B/16 + original DualFeatureMoE + RemoteCLIP attr64 + rot4.
# Pin the current best checkpoint (iteration 29999, validation mIoU 85.3811083).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ ! "${GPU_IDS}" =~ ^[0-9]+,[0-9]+$ || "${#GPUS[@]}" -ne 2 || "${GPUS[0]}" == "${GPUS[1]}" ]]; then
  echo "GPU_IDS must contain two distinct GPU indices" >&2
  exit 2
fi
export TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/clip_vitb_attr64_dual_teacher_dual_feature_moe_rot4_context1e-5_content1e-4_20260912_gpu23/iSAID}"
export CONFIG="${TRAIN_OUTPUT}/config.yaml"
export CHECKPOINT="${CHECKPOINT:-${TRAIN_OUTPUT}/model_0029999.pth}"
export FEATURE_FUSION_TYPE=dual_feature_moe
export ATTRIBUTE_DATABASE="${PROJECT_ROOT}/attributes_text/rskt_seg/iSAID_train_desc_remoteclip_b32_cluster_64_embedding_bank.pth"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_OUTPUT}/eval_model_0029999_all8_gpu01_$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
for dataset in DLRSD iSAID Potsdam Vaihingen UDD5 LoveDA uavid VDD; do
  [[ -f "datasets/${dataset}.json" ]] || { echo "Missing class JSON: ${dataset}" >&2; exit 2; }
done
exec bash scripts/eval_rskt_clip_checkpoint_all8_2gpu.sh iSAID TEST.AUG.ENABLED False "$@"

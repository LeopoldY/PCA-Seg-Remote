#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID} [detectron2 config overrides ...]" >&2
  exit 2
fi

DATASET="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-1,2,3,4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"
RS_DINO_CHECKPOINT="${RS_DINO_CHECKPOINT:-${PROJECT_ROOT}/pretrained/RSIB.pth}"
RESUME="${RESUME:-0}"

case "${DATASET}" in
  DLRSD)
    CONFIG="${CONFIG:-configs/eva_vitb_384_dlrsd_attr64_rs_dino_rskt_protocol.yaml}"
    ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/DLRSD_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr64_rs_dino_rskt_protocol_4gpu_bs2}"
    ;;
  iSAID)
    CONFIG="${CONFIG:-configs/eva_vitb_384_isaid_attr64_rs_dino_rskt_protocol.yaml}"
    ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/iSAID_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr64_rs_dino_rskt_protocol_4gpu_bs2}"
    ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU ids: ${GPU_IDS}" >&2
  exit 2
fi

for path in \
  "${PYTHON_BIN}" \
  "${CONFIG}" \
  "${ATTRIBUTE_DATABASE}" \
  "${DECLIP_CHECKPOINT}" \
  "${RS_DINO_CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
LAUNCH_ARGS=(--config-file "${CONFIG}" --num-gpus 4 --dist-url auto)
if [[ "${RESUME}" == "1" ]]; then
  LAUNCH_ARGS+=(--resume)
fi

echo "Dataset: ${DATASET}"
echo "GPUs: ${GPU_IDS}"
echo "Batch: 2/GPU, 8 global; gradient accumulation: disabled"
echo "Training: 60000 iterations, AdamW, LR 0.0002, cosine schedule"
echo "CLIP initialization: ${DECLIP_CHECKPOINT}"
echo "Architecture: one shared CLIP, four-direction cost aggregation, DualFeatureMoE"
echo "Training-only RS-DINO teacher: ${RS_DINO_CHECKPOINT}"
echo "Inference: CLIP only (no RS-DINO weights required)"
echo "Attribute database: ${ATTRIBUTE_DATABASE} (dataset-specific, 64 centers)"
echo "Resume: ${RESUME}"
echo "Output: ${OUTPUT_DIR}"

OVSISBENCH_DATASETS="${DATASET_ROOT}" \
DETECTRON2_DATASETS="${DATASET_ROOT}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
"${PYTHON_BIN}" train_net.py \
  "${LAUNCH_ARGS[@]}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  MODEL.SEM_SEG_HEAD.CACHE_DIR "${DECLIP_CHECKPOINT}" \
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED True \
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTRIBUTE_DATABASE}" \
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.NUM_CLUSTERS 64 \
  MODEL.SEM_SEG_HEAD.RS_DINO.ENABLED False \
  MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.ENABLED True \
  MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.WEIGHTS "${RS_DINO_CHECKPOINT}" \
  MODEL.SEM_SEG_HEAD.CLIP_ROTATION.ENABLED True \
  MODEL.SEM_SEG_HEAD.FEATURE_FUSION.TYPE dual_feature_moe \
  SOLVER.IMS_PER_BATCH 8 \
  SOLVER.GRAD_ACCUM_STEPS 1 \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID}" >&2
  exit 2
fi

DATASET="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-0}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"
RS_DINO_CHECKPOINT="${RS_DINO_CHECKPOINT:-${PROJECT_ROOT}/pretrained/RSIB.pth}"
REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/RemoteCLIP-ViT-B-32.pt}"

case "${DATASET}" in
  DLRSD)
    CONFIG="configs/eva_vitb_384_dlrsd_attr64_dual_teacher_rskt_protocol.yaml"
    ATTRIBUTE_DATABASE="${PROJECT_ROOT}/attributes_text/rskt_seg/DLRSD_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth"
    TRAIN_DATASET='("DLRSD_train_smoke_sem_seg",)'
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/smoke_dlrsd_dual_teacher_1gpu}"
    ;;
  iSAID)
    CONFIG="configs/eva_vitb_384_isaid_attr64_dual_teacher_rskt_protocol.yaml"
    ATTRIBUTE_DATABASE="${PROJECT_ROOT}/attributes_text/rskt_seg/iSAID_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth"
    TRAIN_DATASET='("iSAID_train_smoke_sem_seg",)'
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/smoke_isaid_dual_teacher_1gpu}"
    ;;
esac

for path in \
  "${PYTHON_BIN}" \
  "${CONFIG}" \
  "${ATTRIBUTE_DATABASE}" \
  "${DECLIP_CHECKPOINT}" \
  "${RS_DINO_CHECKPOINT}" \
  "${REMOTECLIP_CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
echo "Smoke test: ${DATASET}, GPU ${GPU_IDS}, 2 optimizer iterations"

OVSISBENCH_DATASETS="${DATASET_ROOT}" \
DETECTRON2_DATASETS="${DATASET_ROOT}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
"${PYTHON_BIN}" train_net.py \
  --config-file "${CONFIG}" \
  --num-gpus 1 \
  --dist-url auto \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  MODEL.SEM_SEG_HEAD.CACHE_DIR "${DECLIP_CHECKPOINT}" \
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTRIBUTE_DATABASE}" \
  MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.WEIGHTS "${RS_DINO_CHECKPOINT}" \
  MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.WEIGHTS "${REMOTECLIP_CHECKPOINT}" \
  DATASETS.TRAIN "${TRAIN_DATASET}" \
  DATASETS.TEST '()' \
  SOLVER.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER 2 \
  SOLVER.CHECKPOINT_PERIOD 1000 \
  TEST.EVAL_PERIOD 0 \
  DATALOADER.NUM_WORKERS 0

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-1,2,3,4}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
CONFIG="${CONFIG:-configs/eva_vitb_384_isaid_attr64_declip_rskt_protocol.yaml}"
ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/iSAID_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr64_declip_rskt_protocol_4gpu_bs2}"
RESUME="${RESUME:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU ids: ${GPU_IDS}" >&2
  exit 2
fi

for path in "${PYTHON_BIN}" "${CONFIG}" "${ATTRIBUTE_DATABASE}" "${DECLIP_CHECKPOINT}"; do
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

echo "Dataset: iSAID (15 native classes)"
echo "Initialization: DeCLIP EVA02-CLIP-B/16 epoch_latest (epoch 6)"
echo "Attribute database: iSAID-only, 64 centers"
echo "GPUs: ${GPU_IDS}; global batch: 8 (two images/GPU, no accumulation)"
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
  "$@"

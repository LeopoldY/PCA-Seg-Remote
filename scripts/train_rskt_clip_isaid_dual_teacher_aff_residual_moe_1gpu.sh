#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
GPU_ID="${GPU_ID:-4}"
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a single numeric GPU index" >&2
  exit 2
fi
CONFIG="${PROJECT_ROOT}/configs/clip_vitb_384_isaid_attr20_dual_teacher_aff_residual_moe.yaml"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/data6/yc/open-vocab/RSKT-Seg/pretrained/ViT-B-16.pt}"
ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/iSAID_train_desc_remoteclip_b32_cluster_20_embedding_bank.pth}"
RS_DINO_CHECKPOINT="${RS_DINO_CHECKPOINT:-${PROJECT_ROOT}/pretrained/RSIB.pth}"
REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/RemoteCLIP-ViT-B-32.pt}"
# Explicitly select an existing OUTPUT_DIR when resuming.
if [[ "${RESUME:-0}" == "1" && -z "${OUTPUT_DIR:-}" ]]; then
  echo "RESUME=1 requires OUTPUT_DIR pointing to the existing run" >&2
  exit 2
fi
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/clip_vitb_384_isaid_attr20_dual_teacher_aff_residual_moe_remoteclip_attr_content1_context0p1_1gpu_bs8_80k_${RUN_TIMESTAMP}}"
for path in "${PYTHON_BIN}" "${CONFIG}" "${DATASET_ROOT}" "${CLIP_CHECKPOINT}" \
  "${ATTRIBUTE_DATABASE}" "${RS_DINO_CHECKPOINT}" "${REMOTECLIP_CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 2
  fi
done
LAUNCH_ARGS=(--config-file "${CONFIG}" --num-gpus 1 --dist-url auto)
if [[ "${RESUME:-0}" == "1" ]]; then
  LAUNCH_ARGS+=(--resume)
fi
echo "Config: ${CONFIG}"
echo "GPU: ${GPU_ID}; batch: 8; gradient accumulation: 1; max iterations: 80000"
echo "Output: ${OUTPUT_DIR}"
exec env OVSISBENCH_DATASETS="${DATASET_ROOT}" DETECTRON2_DATASETS="${DATASET_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" train_net.py \
  "${LAUNCH_ARGS[@]}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  MODEL.SEM_SEG_HEAD.CACHE_DIR "${CLIP_CHECKPOINT}" \
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTRIBUTE_DATABASE}" \
  MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.WEIGHTS "${RS_DINO_CHECKPOINT}" \
  MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.WEIGHTS "${REMOTECLIP_CHECKPOINT}" \
  SOLVER.IMS_PER_BATCH 8 SOLVER.GRAD_ACCUM_STEPS 1 SOLVER.MAX_ITER 80000 "$@"

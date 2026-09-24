#!/usr/bin/env bash
set -euo pipefail

# Four physical GPUs (1,2,3,4), one image per GPU.
# Official batch-8 schedule: lr=5e-4, max_iter=80k.
# Linear batch scaling to batch 4: lr=2.5e-4, max_iter=160k.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-1,2,3,4}"
DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/mnt/data6/yc/datasets/DETECTRON2_DATASETS}"
CONFIG="${CONFIG:-configs/eva_vitb_384_attr_fusion.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/eva_vitb_384_excel_tse_4gpu_bs1_scaled}"
ATTR_DATABASE="${ATTR_DATABASE:-${PROJECT_ROOT}/attributes_text/excel/ms_coco_desc_eva02_clip_b16_gpt4.0_cluster_224_embedding_bank.pth}"
OPENCLIP_PRETRAINED="${OPENCLIP_PRETRAINED:-${PROJECT_ROOT}/pretrained/EVA02_CLIP_B_psz16_s8B.pt}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
INIT_WEIGHTS="${INIT_WEIGHTS:-}"
RESUME="${RESUME:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU ids; got: ${GPU_IDS}" >&2
  exit 2
fi

if [[ ! -d "${DETECTRON2_DATASETS}" ]]; then
  echo "DETECTRON2_DATASETS directory does not exist: ${DETECTRON2_DATASETS}" >&2
  exit 2
fi

for required_file in \
  "${CONFIG}" \
  "${ATTR_DATABASE}" \
  "${OPENCLIP_PRETRAINED}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file does not exist: ${required_file}" >&2
    exit 2
  fi
done

if [[ -n "${INIT_WEIGHTS}" && ! -f "${INIT_WEIGHTS}" ]]; then
  echo "INIT_WEIGHTS does not exist: ${INIT_WEIGHTS}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

LAUNCH_ARGS=(
  --config-file "${CONFIG}"
  --num-gpus 4
  --dist-url auto
)
if [[ "${RESUME}" == "1" ]]; then
  LAUNCH_ARGS+=(--resume)
fi

CONFIG_OVERRIDES=(
  OUTPUT_DIR "${OUTPUT_DIR}"
  MODEL.SEM_SEG_HEAD.CACHE_DIR "${OPENCLIP_PRETRAINED}"
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED True
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTR_DATABASE}"
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.NUM_CLUSTERS 224
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.TOP_K 0.9
  SOLVER.IMS_PER_BATCH 4
  SOLVER.BASE_LR 0.00025
  SOLVER.MAX_ITER 160000
  SOLVER.CHECKPOINT_PERIOD 10000
  TEST.EVAL_PERIOD 10000
)
if [[ -n "${INIT_WEIGHTS}" ]]; then
  CONFIG_OVERRIDES+=(MODEL.WEIGHTS "${INIT_WEIGHTS}")
fi

echo "GPUs: ${GPU_IDS} (4 processes, 1 image/GPU)"
echo "Dataset root: ${DETECTRON2_DATASETS}"
echo "Effective batch: 4; lr: 0.00025; max_iter: 160000"
echo "Attribute database: ${ATTR_DATABASE}"
echo "Output: ${OUTPUT_DIR}"

DETECTRON2_DATASETS="${DETECTRON2_DATASETS}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
"${PYTHON_BIN}" train_net.py \
  "${LAUNCH_ARGS[@]}" \
  "${CONFIG_OVERRIDES[@]}" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

# Reproduce the original PCA-Seg EVA02-CLIP-B/16 recipe on COCO-Stuff.
# Official: global batch 8, lr 5e-4, 80k iterations.
# This run: 4 GPUs x 1 image/GPU, lr 2.5e-4, 160k iterations. There is no
# gradient accumulation, and the total number of processed images is unchanged.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_IDS="${GPU_IDS:-1,2,3,4}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/mnt/data6/yc/datasets/DETECTRON2_DATASETS}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/eva_vitb_384.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/pcaseg_original_declip_eva_b16_coco_4gpu_bs1_scaled}"
DECLIP_CHECKPOINT_DIR="${DECLIP_CHECKPOINT_DIR:-/mnt/data6/yc/open-vocab/DeCLIP/logs/declip_eva_b16_dinov2b_coco_full_retry1/checkpoints}"
# Model-only export of epoch_latest.pt. Unlike the raw DeCLIP training
# checkpoint, this file is directly loadable with PyTorch 2.6 weights_only.
DECLIP_MODEL_ONLY="${DECLIP_MODEL_ONLY:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-}"
RESUME="${RESUME:-0}"

resolve_declip_checkpoint() {
  if [[ -n "${DECLIP_CHECKPOINT}" ]]; then
    printf '%s\n' "${DECLIP_CHECKPOINT}"
    return
  fi

  local raw_checkpoint=""
  if [[ -f "${DECLIP_CHECKPOINT_DIR}/epoch_latest.pt" ]]; then
    raw_checkpoint="${DECLIP_CHECKPOINT_DIR}/epoch_latest.pt"
  else
    local checkpoints=()
    shopt -s nullglob
    checkpoints=("${DECLIP_CHECKPOINT_DIR}"/epoch_*.pt)
    shopt -u nullglob
    if (( ${#checkpoints[@]} == 0 )); then
      echo "No DeCLIP checkpoint found under: ${DECLIP_CHECKPOINT_DIR}" >&2
      echo "Set DECLIP_CHECKPOINT to an exact model-only checkpoint if needed." >&2
      exit 2
    fi
    raw_checkpoint="$(printf '%s\n' "${checkpoints[@]}" | sort -V | tail -n 1)"
  fi

  if [[ ! -f "${DECLIP_MODEL_ONLY}" || "${raw_checkpoint}" -nt "${DECLIP_MODEL_ONLY}" ]]; then
    echo "Exporting a PyTorch 2.6-compatible model-only DeCLIP checkpoint..." >&2
    "${PYTHON_BIN}" tools/export_declip_eva_checkpoint.py \
      --input "${raw_checkpoint}" \
      --output "${DECLIP_MODEL_ONLY}" >&2
  fi
  printf '%s\n' "${DECLIP_MODEL_ONLY}"
}

DECLIP_CHECKPOINT="$(resolve_declip_checkpoint)"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "Original reproduction expects four comma-separated GPU ids; got: ${GPU_IDS}" >&2
  exit 2
fi

for required_file in "${PYTHON_BIN}" "${CONFIG}" "${DECLIP_CHECKPOINT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required file does not exist: ${required_file}" >&2
    exit 2
  fi
done
if [[ ! -d "${DETECTRON2_DATASETS}" ]]; then
  echo "Dataset root does not exist: ${DETECTRON2_DATASETS}" >&2
  exit 2
fi
if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "RESUME must be 0 or 1; got: ${RESUME}" >&2
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

echo "PCA-Seg recipe: original EVA02-CLIP-B/16 (attribute fusion disabled)"
echo "DeCLIP checkpoint: ${DECLIP_CHECKPOINT}"
echo "GPUs: ${GPU_IDS}; 1 image/GPU; global batch: 4; gradient accumulation: off"
echo "Scaled schedule: lr=0.00025; max_iter=160000"
echo "Dataset root: ${DETECTRON2_DATASETS}"
echo "Output: ${OUTPUT_DIR}"

DETECTRON2_DATASETS="${DETECTRON2_DATASETS}" \
CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
"${PYTHON_BIN}" train_net.py \
  "${LAUNCH_ARGS[@]}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  MODEL.SEM_SEG_HEAD.CACHE_DIR "${DECLIP_CHECKPOINT}" \
  MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED False \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.BASE_LR 0.00025 \
  SOLVER.MAX_ITER 160000 \
  SOLVER.CHECKPOINT_PERIOD 10000 \
  TEST.EVAL_PERIOD 10000 \
  "$@"

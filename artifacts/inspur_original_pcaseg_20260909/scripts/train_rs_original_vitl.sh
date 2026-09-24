#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo "Usage: $0 {DLRSD|iSAID} [config overrides ...]" >&2; exit 2; }
DATASET="$1"; shift
case "$DATASET" in DLRSD) TAG=dlrsd;; iSAID) TAG=isaid;; *) exit 2;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OVSISBENCH_DATASETS="${OVSISBENCH_DATASETS:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/mnt/data6/yc/datasets/DETECTRON2_DATASETS}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/original_openai_vitl336_$TAG}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#GPUS[@]}" == 2 ]] || { echo 'Use exactly 2 GPUs for official global batch 4.' >&2; exit 2; }
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+,[0-9]+$ && "${GPUS[0]}" != "${GPUS[1]}" ]] || { echo 'Specify two different physical GPU indices.' >&2; exit 2; }
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/mnt/data6/yc/open-vocab/PCA-Seg-Remote/pretrained/ViT-L-14-336px.pt}"
[[ -f "$CLIP_CHECKPOINT" ]] || { echo "Missing CLIP checkpoint: $CLIP_CHECKPOINT" >&2; exit 2; }
[[ -x "$PYTHON_BIN" && -d "$OVSISBENCH_DATASETS/${DATASET}_split/train" && -d "$OVSISBENCH_DATASETS/${DATASET}_split/val" ]]
ARGS=(--config-file "configs/vitl_336_${TAG}_rs.yaml" --num-gpus "${#GPUS[@]}" --dist-url auto)
if [[ "${RESUME:-0}" == 1 ]]; then
  [[ -f "$OUTPUT_DIR/last_checkpoint" ]] || { echo 'Resume checkpoint missing' >&2; exit 2; }
  ARGS+=(--resume)
elif [[ "${EVAL_ONLY:-0}" != 1 && -e "$OUTPUT_DIR/last_checkpoint" ]]; then
  echo "Existing checkpoint in $OUTPUT_DIR: use RESUME=1 or a new OUTPUT_DIR." >&2; exit 2
fi
if [[ "${EVAL_ONLY:-0}" == 1 ]]; then
  [[ -f "${WEIGHTS:?Set WEIGHTS to the trained .pth file}" ]]
  ARGS+=(--eval-only)
fi
OPTS=(OUTPUT_DIR "$OUTPUT_DIR" MODEL.SEM_SEG_HEAD.CACHE_DIR "$CLIP_CHECKPOINT")
if [[ "${EVAL_ONLY:-0}" == 1 ]]; then OPTS+=(MODEL.WEIGHTS "$WEIGHTS"); fi
if [[ "${SMOKE:-0}" == 1 ]]; then
  OPTS+=(DATASETS.TRAIN "('${DATASET}_train_smoke_sem_seg',)" DATASETS.TEST "('${DATASET}_val_smoke_sem_seg',)"
    SOLVER.MAX_ITER 2 SOLVER.CHECKPOINT_PERIOD 2 TEST.EVAL_PERIOD 1 DATALOADER.NUM_WORKERS 2)
fi
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  printf 'Command: '
  printf '%q ' "$PYTHON_BIN" train_net.py "${ARGS[@]}" "${OPTS[@]}" "$@"
  printf '\nCUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  exit 0
fi
mkdir -p "$OUTPUT_DIR"
echo "dataset=$DATASET GPUs=$CUDA_VISIBLE_DEVICES batch=4 lr=0.0002 iterations=80000 smoke=${SMOKE:-0} output=$OUTPUT_DIR"
exec "$PYTHON_BIN" train_net.py "${ARGS[@]}" "${OPTS[@]}" "$@"

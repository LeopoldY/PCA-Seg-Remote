#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
REMOTECLIP_CHECKPOINT="${REMOTECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/RemoteCLIP-ViT-L-14.pt}"
[[ -f "${REMOTECLIP_CHECKPOINT}" ]] || { echo "Missing checkpoint: ${REMOTECLIP_CHECKPOINT}" >&2; exit 1; }
for DATASET in DLRSD iSAID; do
  BANK="attributes_text/rskt_seg/${DATASET}_train_desc_remoteclip_l14_cluster_64_embedding_bank.pth"
  if [[ -f "${BANK}" && "${FORCE:-0}" != "1" ]]; then
    echo "Existing bank: ${BANK}"
    continue
  fi
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" "${PYTHON_BIN}" tools/build_attribute_database.py \
    --descriptors-json "attributes_text/${DATASET}_train_descriptors.json" \
    --output "${BANK}" --model-name ViT-L-14 --checkpoint "${REMOTECLIP_CHECKPOINT}" \
    --backend open_clip --device "${DEVICE:-cpu}" --batch-size 128 --num-clusters 64
done

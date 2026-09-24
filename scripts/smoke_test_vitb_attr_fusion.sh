#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

ATTR_DATABASE="${ATTR_DATABASE:-${PROJECT_ROOT}/attributes_text/excel/pascal_voc_desc_eva02_clip_b16_gpt4.0_cluster_112_embedding_bank.pth}"
if [[ -z "${PYTHON_BIN:-}" && -x /opt/miniconda3/envs/yc_d2/bin/python ]]; then
  PYTHON_BIN=/opt/miniconda3/envs/yc_d2/bin/python
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

exec "${PYTHON_BIN}" tools/smoke_test_attribute_fusion.py \
  --attribute-database "${ATTR_DATABASE}" \
  --expected-dim 512 \
  --num-clusters 112 \
  --device "${DEVICE:-cuda}"

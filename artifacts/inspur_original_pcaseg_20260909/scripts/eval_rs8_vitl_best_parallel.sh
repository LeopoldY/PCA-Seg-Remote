#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DLRSD_GPU ISAID_GPU [--source-scope all|val] [--output-root PATH] [--check-only]" >&2
  exit 2
fi
DLRSD_GPU="$1"
ISAID_GPU="$2"
shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}" "$ROOT/scripts/eval_rs8_vitl_best.py" --gpus "$DLRSD_GPU" "$ISAID_GPU" "$@"

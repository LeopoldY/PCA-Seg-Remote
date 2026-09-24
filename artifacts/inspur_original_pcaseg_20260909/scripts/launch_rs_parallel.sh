#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DLRSD_GPUS="${DLRSD_GPUS:-0,1}"
iSAID_GPUS="${ISAID_GPUS:-2,3}"
IFS=',' read -r -a LEFT <<< "$DLRSD_GPUS"
IFS=',' read -r -a RIGHT <<< "$iSAID_GPUS"
for a in "${LEFT[@]}"; do for b in "${RIGHT[@]}"; do
  [[ "$a" != "$b" ]] || { echo "Overlapping GPU $a" >&2; exit 2; }
done; done
RUN_ROOT="${RUN_ROOT:-$ROOT/output/original_openai_vitb16_parallel_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"
for dataset in DLRSD iSAID; do
  pidfile="$RUN_ROOT/$dataset.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "$dataset is already running" >&2; exit 2
  fi
  if [[ "${RESUME:-0}" != 1 && -f "$RUN_ROOT/$dataset/last_checkpoint" ]]; then
    echo "Existing run: use RESUME=1" >&2; exit 2
  fi
done
nohup env GPU_IDS="$DLRSD_GPUS" OUTPUT_DIR="$RUN_ROOT/DLRSD" bash scripts/train_rs_original.sh DLRSD > "$RUN_ROOT/DLRSD.log" 2>&1 < /dev/null &
echo $! > "$RUN_ROOT/DLRSD.pid"
nohup env GPU_IDS="$iSAID_GPUS" OUTPUT_DIR="$RUN_ROOT/iSAID" bash scripts/train_rs_original.sh iSAID > "$RUN_ROOT/iSAID.log" 2>&1 < /dev/null &
echo $! > "$RUN_ROOT/iSAID.pid"
echo "RUN_ROOT=$RUN_ROOT"
cat "$RUN_ROOT/DLRSD.pid" "$RUN_ROOT/iSAID.pid"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ( "$1" != "DLRSD" && "$1" != "iSAID" ) ]]; then
  echo "Usage: $0 {DLRSD|iSAID} [detectron2 config overrides ...]" >&2
  exit 2
fi

TRAIN_DATASET="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"

case "${TRAIN_DATASET}" in
  DLRSD)
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr64_rs_dino_rskt_protocol_4gpu_bs2}"
    ;;
  iSAID)
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr64_rs_dino_rskt_protocol_4gpu_bs2}"
    ;;
esac

METRICS_FILE="${TRAIN_OUTPUT}/metrics.json"
IFS=$'\t' read -r BEST_ITER BEST_MIOU CHECKPOINT < <(
  "${PYTHON_BIN}" - "${METRICS_FILE}" "${TRAIN_OUTPUT}" <<'PY'
import json
import pathlib
import sys

metrics_path = pathlib.Path(sys.argv[1])
train_output = pathlib.Path(sys.argv[2])
if not metrics_path.is_file():
    raise SystemExit(f"Metrics file does not exist: {metrics_path}")

evaluations = []
for line in metrics_path.read_text().splitlines():
    record = json.loads(line)
    if "sem_seg/mIoU" in record and "iteration" in record:
        evaluations.append(
            (int(record["iteration"]), float(record["sem_seg/mIoU"]))
        )
if not evaluations:
    raise SystemExit(f"No sem_seg/mIoU evaluations found in {metrics_path}")

best_iteration, best_miou = max(evaluations, key=lambda item: item[1])
checkpoint = train_output / f"model_{best_iteration:07d}.pth"
if not checkpoint.is_file() and best_iteration == max(
    item[0] for item in evaluations
):
    final_checkpoint = train_output / "model_final.pth"
    if final_checkpoint.is_file():
        checkpoint = final_checkpoint
if not checkpoint.is_file():
    raise SystemExit(
        f"Best checkpoint for iteration {best_iteration} does not exist: "
        f"{checkpoint}"
    )

print(f"{best_iteration}\t{best_miou:.8f}\t{checkpoint}")
PY
)

OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_OUTPUT}/eval_best_miou_all8}"

echo "Training dataset: ${TRAIN_DATASET}"
echo "Best validation iteration: ${BEST_ITER}"
echo "Best validation mIoU: ${BEST_MIOU}"
echo "Best checkpoint: ${CHECKPOINT}"
echo "Output root: ${OUTPUT_ROOT}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec env \
  CHECKPOINT="${CHECKPOINT}" \
  TRAIN_OUTPUT="${TRAIN_OUTPUT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  bash scripts/eval_rs_dino_distill_all8_4gpu.sh \
    "${TRAIN_DATASET}" "$@"

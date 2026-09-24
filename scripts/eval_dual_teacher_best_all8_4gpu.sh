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

GPU_IDS="${GPU_IDS:-0,1,2,3}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/data6/yc/datasets/OVSISBenchDataset}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/yc_d2/bin/python}"
DECLIP_CHECKPOINT="${DECLIP_CHECKPOINT:-${PROJECT_ROOT}/pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt}"
FEATURE_FUSION_TYPE="${FEATURE_FUSION_TYPE:-dual_feature_moe}"

case "${TRAIN_DATASET}" in
  DLRSD)
    CONFIG="${CONFIG:-configs/eva_vitb_384_dlrsd_attr64_dual_teacher_rskt_protocol.yaml}"
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_dlrsd_attr64_dual_teacher_rskt_protocol_4gpu_bs2}"
    ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/DLRSD_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"
    ;;
  iSAID)
    CONFIG="${CONFIG:-configs/eva_vitb_384_isaid_attr64_dual_teacher_rskt_protocol.yaml}"
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_ROOT}/output/eva_vitb_384_isaid_attr64_dual_teacher_rskt_protocol_4gpu_bs2}"
    ATTRIBUTE_DATABASE="${ATTRIBUTE_DATABASE:-${PROJECT_ROOT}/attributes_text/rskt_seg/iSAID_train_desc_declip_eva_b16_cluster_64_embedding_bank.pth}"
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
        evaluations.append((int(record["iteration"]), float(record["sem_seg/mIoU"])))
if not evaluations:
    raise SystemExit(f"No sem_seg/mIoU evaluations found in {metrics_path}")

best_iteration, best_miou = max(evaluations, key=lambda item: item[1])
checkpoint = train_output / f"model_{best_iteration:07d}.pth"
if not checkpoint.is_file():
    if best_iteration == max(item[0] for item in evaluations):
        final_checkpoint = train_output / "model_final.pth"
        if final_checkpoint.is_file():
            checkpoint = final_checkpoint
if not checkpoint.is_file():
    raise SystemExit(
        f"Best checkpoint for iteration {best_iteration} does not exist: {checkpoint}"
    )
print(f"{best_iteration}\t{best_miou:.8f}\t{checkpoint}")
PY
)

OUTPUT_ROOT="${OUTPUT_ROOT:-${TRAIN_OUTPUT}/eval_best_miou_all8}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU ids: ${GPU_IDS}" >&2
  exit 2
fi

for path in \
  "${PYTHON_BIN}" \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  "${ATTRIBUTE_DATABASE}" \
  "${DECLIP_CHECKPOINT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file does not exist: ${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset root does not exist: ${DATASET_ROOT}" >&2
  exit 2
fi

echo "Training dataset: ${TRAIN_DATASET}"
echo "Best validation iteration: ${BEST_ITER}"
echo "Best validation mIoU: ${BEST_MIOU}"
echo "Best checkpoint: ${CHECKPOINT}"
echo "Feature fusion: ${FEATURE_FUSION_TYPE}"
echo "GPUs: ${GPU_IDS}"
echo "Output root: ${OUTPUT_ROOT}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}"
SUMMARY="${OUTPUT_ROOT}/summary.txt"
: > "${SUMMARY}"
failures=()

{
  echo "===== ${TRAIN_DATASET}-trained dual-teacher model: OVSISBench all-8 evaluation ====="
  echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "Best validation iteration: ${BEST_ITER}"
  echo "Best validation mIoU: ${BEST_MIOU}"
  echo "Checkpoint: ${CHECKPOINT}"
  echo "Feature fusion: ${FEATURE_FUSION_TYPE}"
  echo "GPUs: ${GPU_IDS}"
  echo "Inference: student only; both distillation teachers disabled"
} | tee -a "${SUMMARY}"

evaluate() {
  local label="$1"
  local class_json="$2"
  local dataset_name="$3"
  shift 3
  local output_dir="${OUTPUT_ROOT}/${label}"

  mkdir -p "${output_dir}"
  printf '\n===== %s (%s) =====\n' "${label}" "${dataset_name}" | tee -a "${SUMMARY}"

  if OVSISBENCH_DATASETS="${DATASET_ROOT}" \
    DETECTRON2_DATASETS="${DATASET_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    "${PYTHON_BIN}" train_net.py \
      --config-file "${CONFIG}" \
      --num-gpus 4 \
      --dist-url auto \
      --eval-only \
      OUTPUT_DIR "${output_dir}" \
      MODEL.WEIGHTS "${CHECKPOINT}" \
      MODEL.SEM_SEG_HEAD.CACHE_DIR "${DECLIP_CHECKPOINT}" \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED True \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH "${ATTRIBUTE_DATABASE}" \
      MODEL.SEM_SEG_HEAD.ATTR_FUSION.NUM_CLUSTERS 64 \
      MODEL.SEM_SEG_HEAD.RS_DINO.ENABLED False \
      MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.ENABLED False \
      MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.ENABLED False \
      MODEL.SEM_SEG_HEAD.CLIP_ROTATION.ENABLED True \
      MODEL.SEM_SEG_HEAD.CLIP_ROTATION.NUM_DIRECTIONS 4 \
      MODEL.SEM_SEG_HEAD.FEATURE_FUSION.TYPE "${FEATURE_FUSION_TYPE}" \
      MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "${class_json}" \
      DATASETS.TEST "(\"${dataset_name}\",)" \
      TEST.SLIDING_WINDOW False \
      MODEL.SEM_SEG_HEAD.POOLING_SIZES '[1,1]' \
      "$@" \
      2>&1 | tee "${output_dir}/console.log"; then
    grep -E 'Evaluation results for|copypaste:' \
      "${output_dir}/log.txt" | tail -n 4 | tee -a "${SUMMARY}" || true
  else
    failures+=("${label}")
    printf 'FAILED: %s; see %s/console.log\n' \
      "${label}" "${output_dir}" | tee -a "${SUMMARY}"
  fi
}

evaluate DLRSD datasets/DLRSD.json DLRSD_all_sem_seg "$@"
evaluate iSAID datasets/iSAID.json iSAID_all_sem_seg "$@"
evaluate Potsdam datasets/Potsdam.json Potsdam_all_sem_seg "$@"
evaluate Vaihingen datasets/Vaihingen.json Vaihingen_all_sem_seg "$@"
evaluate UDD5 datasets/UDD5.json UDD5_all_sem_seg "$@"
evaluate LoveDA datasets/LoveDA.json LoveDA_all_sem_seg "$@"
evaluate UAVid datasets/uavid.json uavid_all_sem_seg "$@"
evaluate VDD datasets/VDD.json VDD_all_sem_seg "$@"

{
  echo
  echo "===== all-8 evaluation status ====="
  echo "Finished: $(date '+%Y-%m-%d %H:%M:%S %z')"
  if ((${#failures[@]})); then
    echo "Failed datasets: ${failures[*]}"
  else
    echo "All eight OVSISBench evaluations completed successfully."
  fi
} | tee -a "${SUMMARY}"

if ((${#failures[@]})); then
  exit 1
fi

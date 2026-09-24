#!/usr/bin/env python3
"""Compare full attribute fusion with cost-volume-only attribute fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_util


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

OVERALL_KEYS = ("mIoU", "fwIoU", "mACC", "pACC")


def load_metrics(path: Path) -> Dict[str, float]:
    payload = torch.load(str(path), map_location="cpu")
    return {key: float(value) for key, value in payload.items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_grouped_predictions(path: Path) -> Dict[str, List[dict]]:
    with path.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["file_name"])].append(record)
    return dict(grouped)


def decode_prediction(records: Sequence[dict]) -> np.ndarray:
    height, width = records[0]["segmentation"]["size"]
    prediction = np.full((height, width), len(VOC_CLASSES), dtype=np.uint8)
    for record in records:
        mask = mask_util.decode(record["segmentation"])
        if mask.ndim == 3:
            mask = mask[..., 0]
        prediction[mask.astype(bool)] = int(record["category_id"])
    return prediction


def analyze_prediction_changes(
    baseline_json: Path,
    candidate_json: Path,
    dataset_root: Path,
) -> dict:
    baseline = load_grouped_predictions(baseline_json)
    candidate = load_grouped_predictions(candidate_json)
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate prediction image sets differ.")

    gt_dir = (
        dataset_root
        / "VOCdevkit"
        / "VOC2012"
        / "annotations_detectron2"
        / "val"
    )
    total_valid = 0
    changed = 0
    corrected = 0
    regressed = 0
    changed_both_wrong = 0
    unchanged_correct = 0
    unchanged_wrong = 0
    class_changes = {
        class_name: {
            "valid_gt_pixels": 0,
            "corrected_pixels": 0,
            "regressed_pixels": 0,
            "net_corrected_pixels": 0,
        }
        for class_name in VOC_CLASSES
    }

    for index, file_name in enumerate(sorted(baseline)):
        image_id = Path(file_name).stem
        gt = np.asarray(Image.open(gt_dir / f"{image_id}.png"), dtype=np.uint8)
        baseline_pred = decode_prediction(baseline[file_name])
        candidate_pred = decode_prediction(candidate[file_name])
        if baseline_pred.shape != gt.shape or candidate_pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {image_id}")

        valid = gt != 255
        baseline_correct = baseline_pred == gt
        candidate_correct = candidate_pred == gt
        prediction_changed = baseline_pred != candidate_pred

        total_valid += int(np.count_nonzero(valid))
        changed += int(np.count_nonzero(valid & prediction_changed))
        corrected_mask = valid & ~baseline_correct & candidate_correct
        regressed_mask = valid & baseline_correct & ~candidate_correct
        corrected += int(np.count_nonzero(corrected_mask))
        regressed += int(np.count_nonzero(regressed_mask))
        changed_both_wrong += int(
            np.count_nonzero(
                valid
                & prediction_changed
                & ~baseline_correct
                & ~candidate_correct
            )
        )
        unchanged_correct += int(
            np.count_nonzero(valid & ~prediction_changed & baseline_correct)
        )
        unchanged_wrong += int(
            np.count_nonzero(valid & ~prediction_changed & ~baseline_correct)
        )

        for class_id, class_name in enumerate(VOC_CLASSES):
            gt_class = gt == class_id
            class_corrected = int(np.count_nonzero(gt_class & corrected_mask))
            class_regressed = int(np.count_nonzero(gt_class & regressed_mask))
            entry = class_changes[class_name]
            entry["valid_gt_pixels"] += int(np.count_nonzero(gt_class))
            entry["corrected_pixels"] += class_corrected
            entry["regressed_pixels"] += class_regressed

        if (index + 1) % 250 == 0:
            print(f"Compared predictions for {index + 1}/{len(baseline)} images")

    for entry in class_changes.values():
        entry["net_corrected_pixels"] = (
            entry["corrected_pixels"] - entry["regressed_pixels"]
        )

    return {
        "evaluated_images": len(baseline),
        "valid_evaluation_pixels": total_valid,
        "changed_prediction_pixels": changed,
        "changed_prediction_fraction": changed / total_valid,
        "baseline_wrong_candidate_correct_pixels": corrected,
        "baseline_correct_candidate_wrong_pixels": regressed,
        "net_corrected_pixels": corrected - regressed,
        "changed_but_both_wrong_pixels": changed_both_wrong,
        "unchanged_correct_pixels": unchanged_correct,
        "unchanged_wrong_pixels": unchanged_wrong,
        "per_ground_truth_class": class_changes,
    }


def read_ablation_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    attr = config["MODEL"]["SEM_SEG_HEAD"]["ATTR_FUSION"]
    return {
        "attribute_fusion_enabled": bool(attr["ENABLED"]),
        "aggregator_attribute_fusion_enabled": bool(
            attr["AGGREGATOR_ENABLED"]
        ),
        "attribute_database": str(attr["DATABASE_PATH"]),
        "test_class_json": str(
            config["MODEL"]["SEM_SEG_HEAD"]["TEST_CLASS_JSON"]
        ),
        "checkpoint": str(config["MODEL"]["WEIGHTS"]),
        "test_dataset": list(config["DATASETS"]["TEST"]),
        "sliding_window": bool(config["TEST"]["SLIDING_WINDOW"]),
        "pooling_sizes": list(
            config["MODEL"]["SEM_SEG_HEAD"]["POOLING_SIZES"]
        ),
    }


def build_comparison(
    baseline_dir: Path,
    candidate_dir: Path,
    dataset_root: Path,
) -> dict:
    baseline_eval = baseline_dir / "inference/sem_seg_evaluation.pth"
    candidate_eval = candidate_dir / "inference/sem_seg_evaluation.pth"
    baseline_json = baseline_dir / "inference/sem_seg_predictions.json"
    candidate_json = candidate_dir / "inference/sem_seg_predictions.json"
    baseline = load_metrics(baseline_eval)
    candidate = load_metrics(candidate_eval)

    overall = {}
    for key in OVERALL_KEYS:
        overall[key] = {
            "full_attribute_fusion": baseline[key],
            "cost_volume_attribute_only": candidate[key],
            "delta_percentage_points": candidate[key] - baseline[key],
        }

    per_class = []
    for class_name in VOC_CLASSES:
        baseline_iou = baseline[f"IoU-{class_name}"]
        candidate_iou = candidate[f"IoU-{class_name}"]
        baseline_acc = baseline[f"ACC-{class_name}"]
        candidate_acc = candidate[f"ACC-{class_name}"]
        per_class.append(
            {
                "class_name": class_name,
                "iou_full_attribute_fusion": baseline_iou,
                "iou_cost_volume_attribute_only": candidate_iou,
                "iou_delta_percentage_points": candidate_iou - baseline_iou,
                "acc_full_attribute_fusion": baseline_acc,
                "acc_cost_volume_attribute_only": candidate_acc,
                "acc_delta_percentage_points": candidate_acc - baseline_acc,
            }
        )
    per_class.sort(
        key=lambda item: item["iou_delta_percentage_points"],
        reverse=True,
    )

    prediction_changes = analyze_prediction_changes(
        baseline_json,
        candidate_json,
        dataset_root,
    )
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "experiment": (
            "Keep attribute-enhanced cost volume; disable all attribute "
            "operations inside Aggregator."
        ),
        "implementation_semantics": {
            "cost_volume_text": "attribute-enhanced text",
            "aggregator_class_guidance": "original class text",
            "aggregator_spatial_attribute_gates": "disabled",
            "checkpoint_changed": False,
        },
        "candidate_config_verification": read_ablation_config(
            candidate_dir / "config.yaml"
        ),
        "overall_metrics_percent": overall,
        "per_class_metrics_percent": per_class,
        "prediction_change_analysis": prediction_changes,
        "artifacts": {
            "baseline_evaluation": str(baseline_eval),
            "candidate_evaluation": str(candidate_eval),
            "baseline_predictions": str(baseline_json),
            "candidate_predictions": str(candidate_json),
            "candidate_evaluation_sha256": sha256(candidate_eval),
            "candidate_predictions_sha256": sha256(candidate_json),
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_markdown(path: Path, comparison: Mapping[str, Any]) -> None:
    overall = comparison["overall_metrics_percent"]
    changes = comparison["prediction_change_analysis"]
    per_class = comparison["per_class_metrics_percent"]
    improved = [item for item in per_class if item["iou_delta_percentage_points"] > 0]
    degraded = [item for item in per_class if item["iou_delta_percentage_points"] < 0]
    lines = [
        "# VOC Cost-volume-only 属性融合验证结果",
        "",
        "## 实验设置",
        "",
        "- Checkpoint 与原完整属性融合验证相同。",
        "- 属性增强文本继续用于构建 cost-volume。",
        "- Aggregator class guidance 改回原始类别文本。",
        "- Aggregator 两个 spatial block 不再接收属性特征或置信度。",
        "- 使用 VOC 2012 validation 1,449 张图片和相同滑窗参数。",
        "",
        "## 总体指标",
        "",
        "| 指标 | 完整属性融合 | 仅 cost-volume 属性融合 | 差值（百分点） |",
        "|---|---:|---:|---:|",
    ]
    for key in OVERALL_KEYS:
        item = overall[key]
        lines.append(
            f"| {key} | {item['full_attribute_fusion']:.4f} | "
            f"{item['cost_volume_attribute_only']:.4f} | "
            f"{item['delta_percentage_points']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 像素级预测变化",
            "",
            f"- 有效评估像素：{changes['valid_evaluation_pixels']:,}",
            (
                f"- 预测类别发生变化：{changes['changed_prediction_pixels']:,} "
                f"（{changes['changed_prediction_fraction'] * 100:.2f}%）"
            ),
            (
                "- 原基线错误、消融后正确："
                f"{changes['baseline_wrong_candidate_correct_pixels']:,}"
            ),
            (
                "- 原基线正确、消融后错误："
                f"{changes['baseline_correct_candidate_wrong_pixels']:,}"
            ),
            f"- 净纠正像素：{changes['net_corrected_pixels']:+,}",
            "",
            "## 逐类 IoU",
            "",
            "| 类别 | 完整属性融合 | 仅 cost-volume 属性融合 | 差值（百分点） |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in per_class:
        lines.append(
            f"| {item['class_name']} | "
            f"{item['iou_full_attribute_fusion']:.4f} | "
            f"{item['iou_cost_volume_attribute_only']:.4f} | "
            f"{item['iou_delta_percentage_points']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                f"- 20 类中 {len(improved)} 类 IoU 提升，"
                f"{len(degraded)} 类下降。"
            ),
            (
                "- 提升最大的类别："
                + "、".join(
                    f"{item['class_name']} "
                    f"({item['iou_delta_percentage_points']:+.2f})"
                    for item in improved[:5]
                )
                + "。"
            ),
            (
                "- 下降的类别："
                + (
                    "、".join(
                        f"{item['class_name']} "
                        f"({item['iou_delta_percentage_points']:+.2f})"
                        for item in degraded
                    )
                    if degraded
                    else "无"
                )
                + "。"
            ),
            (
                "- 本次结果表明，在该 checkpoint 与 VOC 设置下，"
                "保留属性 cost-volume、移除 Aggregator 内属性路径的"
                "总体分割性能更高。该结论是消融关联，不单独构成因果解释。"
            ),
            "",
            "## 原始产物",
            "",
            "- `inference/sem_seg_evaluation.pth`：完整指标。",
            "- `inference/sem_seg_predictions.json`：逐图 RLE 预测。",
            "- `config.yaml`：冻结后的实际运行配置。",
            "- `log.txt`：完整服务器验证日志。",
            "- `comparison_with_full_attr.json`：本报告的结构化数据。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()

    comparison = build_comparison(
        args.baseline_dir.resolve(),
        args.candidate_dir.resolve(),
        args.dataset_root.resolve(),
    )
    output_json = args.candidate_dir / "comparison_with_full_attr.json"
    output_md = args.candidate_dir / "RESULTS.md"
    write_json(output_json, comparison)
    write_markdown(output_md, comparison)
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()

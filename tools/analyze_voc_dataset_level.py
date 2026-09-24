#!/usr/bin/env python3
"""Dataset-level prediction and attribute-retrieval analysis for VOC-20.

This script combines the stored Detectron2 semantic predictions with an exact
attribute-enhanced sliding-window inference pass.  It reports all 20 classes,
all validation images, dominant confusion directions, and image-conditioned
attribute retrieval behavior without retaining large per-image feature maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualize_voc_confusions import SlidingFeatureCapture, build_predictor  # noqa: E402
from visualize_voc_feature_ablation import (  # noqa: E402
    VOC_CLASSES,
    decode_prediction,
    group_predictions,
    write_json,
)


TYPE_NAMES = {
    0: "color",
    1: "material",
    2: "shape",
    3: "texture",
    4: "part",
    5: "state",
    6: "relation",
    7: "other",
}


def finite_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or a[valid].std() == 0 or b[valid].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gt_path_for(image_path: Path) -> Path:
    return image_path.parents[1] / "annotations_detectron2" / "val" / f"{image_path.stem}.png"


def draw_title(canvas: np.ndarray, title: str, subtitle: str = "") -> None:
    cv2.putText(canvas, title, (34, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.02, (25, 25, 25), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (34, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)


def save_grouped_bar(
    path: Path,
    title: str,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], tuple[int, int, int]]],
    ylabel: str,
    value_scale: float = 1.0,
) -> None:
    width, height = 2300, 980
    left, right, top, bottom = 130, 40, 105, 210
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_title(canvas, title, ylabel)
    values = np.asarray([s[1] for s in series], dtype=np.float64) * value_scale
    finite = values[np.isfinite(values)]
    vmin = min(0.0, float(finite.min())) if finite.size else 0.0
    vmax = max(1.0, float(finite.max())) if finite.size else 1.0
    if vmin < 0:
        span = vmax - vmin
        vmin -= 0.08 * span
        vmax += 0.08 * span
    else:
        vmax *= 1.12
    chart_h = height - top - bottom
    chart_w = width - left - right

    def y_of(value: float) -> int:
        return int(top + (vmax - value) / max(vmax - vmin, 1e-9) * chart_h)

    ticks = 6
    for tick in range(ticks + 1):
        value = vmin + (vmax - vmin) * tick / ticks
        y = y_of(value)
        cv2.line(canvas, (left, y), (width - right, y), (225, 225, 225), 1)
        cv2.putText(canvas, f"{value:.2f}", (18, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (70, 70, 70), 1, cv2.LINE_AA)
    zero_y = y_of(0.0)
    cv2.line(canvas, (left, zero_y), (width - right, zero_y), (80, 80, 80), 2)
    group_w = chart_w / len(labels)
    usable = group_w * 0.76
    bar_w = max(3, int(usable / max(1, len(series))))
    for i, label in enumerate(labels):
        center = left + (i + 0.5) * group_w
        for j, (_, vals, color) in enumerate(series):
            value = float(vals[i]) * value_scale
            if not math.isfinite(value):
                continue
            x1 = int(center - usable / 2 + j * bar_w)
            x2 = x1 + bar_w - 2
            y = y_of(value)
            cv2.rectangle(canvas, (x1, min(y, zero_y)), (x2, max(y, zero_y)), color, -1)
        org = (int(center - 2), height - bottom + 18)
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.43, (35, 35, 35), 1, cv2.LINE_AA)
    legend_spacing = chart_w / max(1, len(series))
    for legend_id, (name, _, color) in enumerate(series):
        legend_x = int(left + legend_id * legend_spacing)
        cv2.rectangle(canvas, (legend_x, height - 66), (legend_x + 26, height - 44), color, -1)
        cv2.putText(canvas, name, (legend_x + 34, height - 47), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def save_heatmap(path: Path, title: str, labels: Sequence[str], matrix: np.ndarray, fmt: str = ".2f") -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    n = len(labels)
    cell = 72
    left, top, right, bottom = 210, 110, 35, 170
    canvas = np.full((top + n * cell + bottom, left + n * cell + right, 3), 255, dtype=np.uint8)
    draw_title(canvas, title)
    finite = matrix[np.isfinite(matrix)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if low < 0 < high:
        bound = max(abs(low), abs(high))
        low, high = -bound, bound
    for row in range(n):
        for col in range(n):
            value = matrix[row, col]
            norm = 0.5 if high == low else (value - low) / (high - low)
            if low < 0 < high:
                if norm < 0.5:
                    t = norm * 2
                    color = (255, int(255 * t), int(255 * t))
                else:
                    t = (norm - 0.5) * 2
                    color = (int(255 * (1 - t)), int(255 * (1 - t)), 255)
            else:
                color = (255 - int(185 * norm), 255 - int(105 * norm), 255 - int(20 * norm))
            x1, y1 = left + col * cell, top + row * cell
            cv2.rectangle(canvas, (x1, y1), (x1 + cell, y1 + cell), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x1 + cell, y1 + cell), (235, 235, 235), 1)
            text = format(value, fmt) if math.isfinite(value) else "NA"
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
            cv2.putText(canvas, text, (x1 + (cell - tw) // 2, y1 + cell // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, labels[row], (8, top + row * cell + cell // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
    for col, label in enumerate(labels):
        # Rotate column labels vertically so all 20 class names remain readable
        # without overlapping neighboring columns.
        text_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1
        )
        patch = np.full(
            (text_size[1] + baseline + 8, text_size[0] + 8, 3),
            255,
            dtype=np.uint8,
        )
        cv2.putText(
            patch,
            label,
            (4, text_size[1] + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
        x = left + col * cell + (cell - patch.shape[1]) // 2
        y = top + n * cell + 8
        canvas[y : y + patch.shape[0], x : x + patch.shape[1]] = patch
    cv2.putText(canvas, "columns", (left + n * cell // 2 - 35, canvas.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def entropy_normalized(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size <= 1:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum() / np.log(counts.size))


def main(args: argparse.Namespace) -> None:
    import torch

    output_dir = Path(args.output_dir).resolve()
    charts_dir = output_dir / "charts"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = group_predictions(Path(args.predictions).resolve())
    image_items = sorted(grouped.items(), key=lambda item: Path(item[0]).stem)
    if args.max_images is not None:
        image_items = image_items[: int(args.max_images)]
    class_count = len(VOC_CLASSES)
    confusion = np.zeros((class_count, class_count + 1), dtype=np.int64)
    image_records: list[dict[str, Any]] = []
    class_image_records: list[list[dict[str, Any]]] = [[] for _ in range(class_count)]

    print(f"Phase 1/2: decoding {len(image_items)} prediction images", flush=True)
    for image_index, (image_name, records) in enumerate(image_items, start=1):
        image_path = Path(image_name)
        gt_path = gt_path_for(image_path)
        gt = np.asarray(Image.open(gt_path), dtype=np.uint8)
        pred = decode_prediction(records, gt.shape)
        valid = gt != 255
        packed = gt[valid].astype(np.int64) * (class_count + 1) + pred[valid].astype(np.int64)
        confusion += np.bincount(packed, minlength=class_count * (class_count + 1)).reshape(class_count, class_count + 1)
        valid_count = int(valid.sum())
        correct = int((gt[valid] == pred[valid]).sum())
        present_ids = np.unique(gt[valid]).astype(int).tolist()
        present_ious = []
        for class_id in range(class_count):
            gt_mask = gt == class_id
            pred_mask = (pred == class_id) & valid
            gt_pixels = int(gt_mask.sum())
            pred_pixels = int(pred_mask.sum())
            tp = int((gt_mask & pred_mask).sum())
            fp = pred_pixels - tp
            fn = gt_pixels - tp
            union = tp + fp + fn
            rec = {
                "image_id": image_path.stem,
                "class_id": class_id,
                "class_name": VOC_CLASSES[class_id],
                "present": int(gt_pixels > 0),
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "iou": safe_div(tp, union),
                "recall": safe_div(tp, gt_pixels),
                "precision": safe_div(tp, pred_pixels),
                "gt_fraction": safe_div(gt_pixels, valid_count),
                "pred_fraction": safe_div(pred_pixels, valid_count),
            }
            class_image_records[class_id].append(rec)
            if gt_pixels > 0:
                present_ious.append(rec["iou"])
        image_records.append({
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "ground_truth_path": str(gt_path),
            "valid_pixels": valid_count,
            "pixel_accuracy": safe_div(correct, valid_count),
            "present_class_count": len(present_ids),
            "mean_present_class_iou": float(np.nanmean(present_ious)),
        })
        if image_index % 250 == 0:
            print(f"  decoded {image_index}/{len(image_items)}", flush=True)

    class_rows: list[dict[str, Any]] = []
    dominant_wrong_ids: list[int] = []
    pair_rows: list[dict[str, Any]] = []
    for gt_id, class_name in enumerate(VOC_CLASSES):
        row = confusion[gt_id]
        gt_pixels = int(row.sum())
        tp = int(row[gt_id])
        pred_pixels = int(confusion[:, gt_id].sum())
        fp, fn = pred_pixels - tp, gt_pixels - tp
        union = tp + fp + fn
        wrong = row[:class_count].copy()
        wrong[gt_id] = 0
        wrong_id = int(wrong.argmax())
        dominant_wrong_ids.append(wrong_id)
        class_rows.append({
            "class_id": gt_id,
            "class_name": class_name,
            "gt_pixels": gt_pixels,
            "pred_pixels": pred_pixels,
            "pred_to_gt_ratio": safe_div(pred_pixels, gt_pixels),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": safe_div(tp, pred_pixels),
            "recall": safe_div(tp, gt_pixels),
            "iou": safe_div(tp, union),
            "dominant_wrong_class": VOC_CLASSES[wrong_id],
            "dominant_wrong_pixels": int(wrong[wrong_id]),
            "dominant_wrong_gt_fraction": safe_div(wrong[wrong_id], gt_pixels),
            "present_image_count": int(sum(r["present"] for r in class_image_records[gt_id])),
        })
        for pred_id, count in enumerate(row[:class_count]):
            if pred_id != gt_id and count:
                pair_rows.append({
                    "gt_class_id": gt_id,
                    "gt_class": class_name,
                    "pred_class_id": pred_id,
                    "pred_class": VOC_CLASSES[pred_id],
                    "pixels": int(count),
                    "fraction_of_gt": safe_div(count, gt_pixels),
                })
    pair_rows.sort(key=lambda row: row["pixels"], reverse=True)

    predictor = build_predictor(args)
    model = predictor.model
    aggregator = model.sem_seg_head.predictor.transformer
    capture = SlidingFeatureCapture(aggregator)
    # The standard visualization capture is extended here with the compact
    # retrieved feature vector, but no spatial tensors are retained.
    original_capture = capture._capture_attributes
    def capture_attributes(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        original_capture(module, inputs, output)
        capture.cache["retrieval"]["features"] = output.attr_features.detach().float().cpu().numpy()
    capture._handles[-1].remove()
    capture._handles[-1] = aggregator.attribute_adapter.register_forward_hook(capture_attributes)

    database = torch.load(str(Path(args.attribute_database).resolve()), map_location="cpu")
    phrases = list(database["representative_phrases"])
    type_ids = database["attribute_type_ids"].cpu().numpy().astype(int)
    cluster_count = len(phrases)
    feature_dim = int(database["embeddings"].shape[1])
    confidence_all: list[list[float]] = [[] for _ in range(class_count)]
    confidence_present: list[list[float]] = [[] for _ in range(class_count)]
    confidence_absent: list[list[float]] = [[] for _ in range(class_count)]
    feature_sum = np.zeros((class_count, feature_dim), dtype=np.float64)
    feature_samples = np.zeros(class_count, dtype=np.int64)
    retrieval_counts = np.zeros((class_count, cluster_count), dtype=np.int64)
    retrieval_score_sum = np.zeros((class_count, cluster_count), dtype=np.float64)
    retrieval_type_counts = np.zeros((class_count, max(TYPE_NAMES) + 1), dtype=np.int64)
    valid_selection_count = np.zeros(class_count, dtype=np.int64)
    attr_image_rows: list[dict[str, Any]] = []

    print(f"Phase 2/2: exact sliding-window attribute inference on {len(image_items)} images", flush=True)
    try:
        for image_index, (image_name, records) in enumerate(image_items, start=1):
            image_path = Path(image_name)
            gt = np.asarray(Image.open(gt_path_for(image_path)), dtype=np.uint8)
            valid = gt != 255
            pred = decode_prediction(records, gt.shape)
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(image_path)
            capture.clear()
            with torch.no_grad():
                predictor(image_bgr)
            retrieval = capture.cache["retrieval"]
            conf = retrieval["confidence"].mean(axis=0)
            features = retrieval["features"].mean(axis=0)
            selected_indices = retrieval["indices"]
            selected_scores = retrieval["scores"]
            selected_mask = retrieval["mask"].astype(bool)
            for class_id in range(class_count):
                present = bool(np.any(gt == class_id))
                confidence = float(conf[class_id])
                confidence_all[class_id].append(confidence)
                (confidence_present[class_id] if present else confidence_absent[class_id]).append(confidence)
                feature_sum[class_id] += features[class_id]
                feature_samples[class_id] += 1
                ids = selected_indices[:, class_id][selected_mask[:, class_id]].astype(int)
                scores = selected_scores[:, class_id][selected_mask[:, class_id]].astype(float)
                valid_selection_count[class_id] += len(ids)
                for cluster_id, score in zip(ids, scores):
                    retrieval_counts[class_id, cluster_id] += 1
                    retrieval_score_sum[class_id, cluster_id] += score
                    retrieval_type_counts[class_id, type_ids[cluster_id]] += 1
                metrics = class_image_records[class_id][image_index - 1]
                pair_error_pixels = int(np.count_nonzero((gt == class_id) & (pred == dominant_wrong_ids[class_id])))
                attr_image_rows.append({
                    "image_id": image_path.stem,
                    "class_id": class_id,
                    "class_name": VOC_CLASSES[class_id],
                    "present": int(present),
                    "attribute_confidence": confidence,
                    "iou": metrics["iou"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "gt_fraction": metrics["gt_fraction"],
                    "pred_fraction": metrics["pred_fraction"],
                    "dominant_wrong_class": VOC_CLASSES[dominant_wrong_ids[class_id]],
                    "dominant_pair_error_fraction": safe_div(pair_error_pixels, metrics["gt_pixels"]),
                    "wrong_attribute_confidence": float(conf[dominant_wrong_ids[class_id]]),
                    "wrong_minus_gt_confidence": float(conf[dominant_wrong_ids[class_id]] - confidence),
                })
            if image_index % 50 == 0 or image_index == len(image_items):
                print(f"  inferred {image_index}/{len(image_items)}", flush=True)
    finally:
        capture.close()

    attr_rows: list[dict[str, Any]] = []
    phrase_rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(VOC_CLASSES):
        present_rows = [r for r in attr_image_rows if r["class_id"] == class_id and r["present"]]
        counts = retrieval_counts[class_id]
        top_ids = np.argsort(counts)[::-1][:10]
        total = int(counts.sum())
        top10 = int(counts[top_ids].sum())
        type_total = int(retrieval_type_counts[class_id].sum())
        for rank, cluster_id in enumerate(top_ids, start=1):
            count = int(counts[cluster_id])
            if count == 0:
                continue
            phrase_rows.append({
                "class_id": class_id,
                "class_name": class_name,
                "rank": rank,
                "cluster_id": int(cluster_id),
                "representative_phrase": phrases[cluster_id],
                "attribute_type": TYPE_NAMES.get(int(type_ids[cluster_id]), str(type_ids[cluster_id])),
                "retrieval_count_across_views": count,
                "retrieval_fraction": safe_div(count, total),
                "mean_selected_score": safe_div(retrieval_score_sum[class_id, cluster_id], count),
            })
        gt_conf = np.asarray([r["attribute_confidence"] for r in present_rows], dtype=float)
        wrong_conf = np.asarray([r["wrong_attribute_confidence"] for r in present_rows], dtype=float)
        attr_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence_all_mean": float(np.mean(confidence_all[class_id])),
            "confidence_all_std": float(np.std(confidence_all[class_id])),
            "confidence_present_mean": float(np.mean(confidence_present[class_id])),
            "confidence_absent_mean": float(np.mean(confidence_absent[class_id])),
            "present_minus_absent": float(np.mean(confidence_present[class_id]) - np.mean(confidence_absent[class_id])),
            "confidence_iou_correlation": safe_corr([r["attribute_confidence"] for r in present_rows], [r["iou"] for r in present_rows]),
            "confidence_recall_correlation": safe_corr([r["attribute_confidence"] for r in present_rows], [r["recall"] for r in present_rows]),
            "confidence_pred_area_correlation": safe_corr([r["attribute_confidence"] for r in present_rows], [r["pred_fraction"] for r in present_rows]),
            "confidence_gt_area_correlation": safe_corr([r["attribute_confidence"] for r in present_rows], [r["gt_fraction"] for r in present_rows]),
            "dominant_wrong_class": VOC_CLASSES[dominant_wrong_ids[class_id]],
            "dominant_wrong_confidence_mean": float(wrong_conf.mean()),
            "wrong_confidence_exceeds_gt_fraction": float(np.mean(wrong_conf > gt_conf)),
            "confidence_margin_pair_error_correlation": safe_corr([r["wrong_minus_gt_confidence"] for r in present_rows], [r["dominant_pair_error_fraction"] for r in present_rows]),
            "retrieved_items_across_views": total,
            "unique_retrieved_clusters": int(np.count_nonzero(counts)),
            "top10_retrieval_concentration": safe_div(top10, total),
            "retrieval_entropy_normalized": entropy_normalized(counts),
            **{
                f"{type_name}_fraction": safe_div(
                    retrieval_type_counts[class_id, type_id], type_total
                )
                for type_id, type_name in TYPE_NAMES.items()
            },
        })

    mean_features = feature_sum / np.maximum(feature_samples[:, None], 1)
    norm = np.linalg.norm(mean_features, axis=1, keepdims=True)
    feature_similarity = (mean_features / np.maximum(norm, 1e-12)) @ (mean_features / np.maximum(norm, 1e-12)).T
    similarity_rows = []
    for i in range(class_count):
        for j in range(i + 1, class_count):
            similarity_rows.append({"class_a": VOC_CLASSES[i], "class_b": VOC_CLASSES[j], "cosine_similarity": float(feature_similarity[i, j])})
    similarity_rows.sort(key=lambda row: row["cosine_similarity"], reverse=True)

    metrics_by_name = {r["class_name"]: r for r in class_rows}
    attrs_by_name = {r["class_name"]: r for r in attr_rows}
    merged_rows = [{**metrics_by_name[name], **attrs_by_name[name]} for name in VOC_CLASSES]
    float_fields = {
        "precision", "recall", "iou", "pred_to_gt_ratio", "dominant_wrong_gt_fraction",
        "confidence_all_mean", "confidence_all_std", "confidence_present_mean", "confidence_absent_mean",
        "present_minus_absent", "confidence_iou_correlation", "confidence_recall_correlation",
        "confidence_pred_area_correlation", "confidence_gt_area_correlation", "dominant_wrong_confidence_mean",
        "wrong_confidence_exceeds_gt_fraction", "confidence_margin_pair_error_correlation",
        "top10_retrieval_concentration", "retrieval_entropy_normalized",
        *[f"{name}_fraction" for name in TYPE_NAMES.values()],
    }
    for row in merged_rows:
        for field in float_fields:
            if field in row:
                row[field] = finite_float(row[field])
    for row in attr_image_rows:
        for field in ("attribute_confidence", "iou", "recall", "precision", "gt_fraction", "pred_fraction", "dominant_pair_error_fraction", "wrong_attribute_confidence", "wrong_minus_gt_confidence"):
            row[field] = finite_float(row[field])

    write_csv(tables_dir / "per_class_complete_metrics.csv", merged_rows, list(merged_rows[0].keys()))
    write_csv(tables_dir / "all_confusion_pairs.csv", pair_rows, list(pair_rows[0].keys()))
    write_csv(tables_dir / "per_image_summary.csv", image_records, list(image_records[0].keys()))
    write_csv(tables_dir / "per_image_class_attribute_metrics.csv", attr_image_rows, list(attr_image_rows[0].keys()))
    write_csv(tables_dir / "top_retrieved_attributes_per_class.csv", phrase_rows, list(phrase_rows[0].keys()))
    write_csv(tables_dir / "attribute_feature_similarity_pairs.csv", similarity_rows, list(similarity_rows[0].keys()))
    write_csv(tables_dir / "confusion_matrix_counts.csv", [dict(gt_class=VOC_CLASSES[i], **{VOC_CLASSES[j]: int(confusion[i, j]) for j in range(class_count)}, uncovered=int(confusion[i, class_count])) for i in range(class_count)], ["gt_class", *VOC_CLASSES, "uncovered"])
    np.savez_compressed(output_dir / "dataset_level_arrays.npz", confusion=confusion, mean_attribute_features=mean_features, attribute_feature_similarity=feature_similarity, retrieval_counts=retrieval_counts, retrieval_type_counts=retrieval_type_counts)

    label_short = [name[:10] for name in VOC_CLASSES]
    save_grouped_bar(charts_dir / "all_class_segmentation_metrics.png", "VOC-20 per-class segmentation metrics", label_short, [("Precision", [r["precision"] for r in class_rows], (220, 150, 55)), ("Recall", [r["recall"] for r in class_rows], (70, 170, 90)), ("IoU", [r["iou"] for r in class_rows], (60, 90, 220))], "fraction (0-1)")
    save_grouped_bar(charts_dir / "prediction_volume_ratio.png", "Prediction volume relative to GT", label_short, [("Predicted / GT pixels", [r["pred_to_gt_ratio"] for r in class_rows], (70, 130, 220))], "ratio; 1.0 means matched volume")
    save_grouped_bar(charts_dir / "attribute_confidence_presence.png", "Attribute confidence: GT present versus absent", label_short, [("GT present", [r["confidence_present_mean"] for r in attr_rows], (70, 170, 90)), ("GT absent", [r["confidence_absent_mean"] for r in attr_rows], (160, 160, 160))], "mean over 5 validation views")
    save_grouped_bar(charts_dir / "attribute_confidence_correlations.png", "Attribute confidence correlations on GT-present images", label_short, [("with IoU", [r["confidence_iou_correlation"] for r in attr_rows], (60, 90, 220)), ("with recall", [r["confidence_recall_correlation"] for r in attr_rows], (70, 170, 90)), ("with predicted area", [r["confidence_pred_area_correlation"] for r in attr_rows], (220, 150, 55))], "Pearson r")
    save_grouped_bar(charts_dir / "dominant_confusion_attribute_competition.png", "Attribute confidence competition for each dominant confusion", label_short, [("GT class confidence", [r["confidence_present_mean"] for r in attr_rows], (70, 170, 90)), ("dominant wrong class confidence", [r["dominant_wrong_confidence_mean"] for r in attr_rows], (60, 90, 220)), ("P(wrong > GT)", [r["wrong_confidence_exceeds_gt_fraction"] for r in attr_rows], (80, 180, 210))], "confidence / fraction")
    type_colors = [(220, 150, 55), (190, 110, 190), (60, 160, 210), (90, 190, 190), (70, 170, 90), (180, 130, 80), (60, 90, 220), (160, 160, 160)]
    save_grouped_bar(
        charts_dir / "attribute_type_distribution.png",
        "Retrieved attribute-type distribution",
        label_short,
        [
            (name, [r[f"{name}_fraction"] for r in attr_rows], type_colors[type_id])
            for type_id, name in TYPE_NAMES.items()
        ],
        "fraction of valid retrieved slots",
    )
    save_heatmap(charts_dir / "attribute_feature_similarity_matrix.png", "Mean recalled attribute-feature cosine similarity", label_short, feature_similarity)
    row_norm = confusion[:, :class_count] / np.maximum(confusion[:, :class_count].sum(axis=1, keepdims=True), 1)
    save_heatmap(charts_dir / "confusion_matrix_row_normalized.png", "VOC-20 confusion matrix (row-normalized; GT rows, prediction columns)", label_short, row_norm)

    summary = {
        "dataset": "PASCAL VOC 2012 validation, 20 foreground classes",
        "evaluated_images": len(image_items),
        "valid_foreground_pixels": int(confusion.sum()),
        "prediction_metrics": {
            "mIoU_percent": float(np.mean([r["iou"] for r in class_rows]) * 100),
            "mACC_percent": float(np.mean([r["recall"] for r in class_rows]) * 100),
            "pACC_percent": float(np.trace(confusion[:, :class_count]) / confusion.sum() * 100),
        },
        "per_class": merged_rows,
        "top_confusion_pairs": pair_rows[:50],
        "top_attribute_feature_similarity_pairs": similarity_rows[:30],
        "worst_images_by_pixel_accuracy": sorted(image_records, key=lambda row: row["pixel_accuracy"])[:30],
        "best_images_by_pixel_accuracy": sorted(image_records, key=lambda row: row["pixel_accuracy"], reverse=True)[:30],
        "retrieval": {
            "database_clusters": cluster_count,
            "feature_dimension": feature_dim,
            "views_per_image": 5,
            "top_attributes_per_class": phrase_rows,
        },
        "methodology": {
            "prediction_source": str(Path(args.predictions).resolve()),
            "attribute_inference": "Exact configured 4 local windows plus 1 global view; confidence and features averaged over five views.",
            "presence_definition": "The foreground class has at least one valid GT pixel in the validation image.",
            "correlations": "Pearson correlation computed only on GT-present images for the class.",
        },
    }
    write_json(output_dir / "dataset_level_summary.json", summary)
    print(f"Complete dataset-level analysis written to {output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--openclip-pretrained", required=True)
    parser.add_argument("--attribute-database", required=True)
    parser.add_argument("--class-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test limit")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())

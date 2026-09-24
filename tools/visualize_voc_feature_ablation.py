#!/usr/bin/env python3
"""Analyze VOC errors and visualize PCA-Seg attribute-fusion feature maps.

The script has two phases:

1. ``analyze`` decodes Detectron2 semantic predictions, computes a ground-truth
   class -> predicted class confusion matrix, ranks classes by false-negative
   pixels, and selects representative worst images.
2. ``visualize`` loads the trained attribute-fusion checkpoint once and runs a
   controlled inference-time ablation on the selected images. Forward hooks
   capture the raw cost volume and the output of the final AggregatorLayer.

The attribute-disabled pass uses exactly the same checkpoint and model instance;
only ``Aggregator.use_attribute_fusion`` is toggled. This isolates the attribute
path without changing the learned non-attribute aggregation weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

# Bright, visually distinct RGB colors. Ignore pixels remain transparent.
VOC_COLORS = np.asarray(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
        [128, 128, 128],
    ],
    dtype=np.uint8,
)


def json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON serialize {type(value)!r}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            default=json_scalar,
        )
        stream.write("\n")


def group_predictions(prediction_json: Path) -> Dict[str, List[dict]]:
    with prediction_json.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["file_name"])].append(record)
    return dict(grouped)


def decode_prediction(records: Sequence[dict], shape: Sequence[int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    # 20 denotes an uncovered/invalid predicted pixel. Normal evaluator output
    # covers every pixel, but retaining the sentinel makes data issues explicit.
    pred = np.full((height, width), len(VOC_CLASSES), dtype=np.uint8)
    for record in records:
        category_id = int(record["category_id"])
        decoded = mask_util.decode(record["segmentation"])
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        if decoded.shape != pred.shape:
            raise ValueError(
                f"Prediction shape {decoded.shape} does not match GT {pred.shape}"
            )
        pred[decoded.astype(bool)] = category_id
    return pred


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def per_image_class_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    class_id: int,
) -> dict:
    valid = gt != 255
    gt_mask = gt == class_id
    pred_mask = (pred == class_id) & valid
    tp = int(np.count_nonzero(gt_mask & pred_mask))
    gt_pixels = int(np.count_nonzero(gt_mask))
    pred_pixels = int(np.count_nonzero(pred_mask))
    fn = gt_pixels - tp
    fp = pred_pixels - tp
    union = tp + fn + fp
    return {
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "true_positive_pixels": tp,
        "false_negative_pixels": fn,
        "false_positive_pixels": fp,
        "class_accuracy": safe_ratio(tp, gt_pixels),
        "iou": safe_ratio(tp, union),
    }


def analyze_predictions(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    prediction_json = Path(args.predictions).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    gt_dir = (
        dataset_root
        / "VOCdevkit"
        / "VOC2012"
        / "annotations_detectron2"
        / "val"
    )
    grouped = group_predictions(prediction_json)

    # Rows are GT classes, columns are predicted classes plus one sentinel.
    confusion = np.zeros(
        (len(VOC_CLASSES), len(VOC_CLASSES) + 1),
        dtype=np.int64,
    )
    image_metrics: Dict[int, List[dict]] = defaultdict(list)
    uncovered_valid_pixels = 0

    for image_index, (image_path_str, records) in enumerate(
        sorted(grouped.items())
    ):
        image_path = Path(image_path_str)
        image_id = image_path.stem
        gt_path = gt_dir / f"{image_id}.png"
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing VOC ground truth: {gt_path}")
        gt = np.asarray(Image.open(gt_path), dtype=np.uint8)
        pred = decode_prediction(records, gt.shape)
        valid = gt != 255
        uncovered_valid_pixels += int(
            np.count_nonzero(valid & (pred == len(VOC_CLASSES)))
        )

        encoded = (
            (len(VOC_CLASSES) + 1) * gt[valid].astype(np.int64)
            + pred[valid].astype(np.int64)
        )
        confusion += np.bincount(
            encoded,
            minlength=len(VOC_CLASSES) * (len(VOC_CLASSES) + 1),
        ).reshape(confusion.shape)

        present_classes = np.unique(gt[valid])
        for class_id_value in present_classes:
            class_id = int(class_id_value)
            metrics = per_image_class_metrics(gt, pred, class_id)
            metrics.update(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "ground_truth_path": str(gt_path),
                    "height": int(gt.shape[0]),
                    "width": int(gt.shape[1]),
                }
            )
            image_metrics[class_id].append(metrics)

        if (image_index + 1) % 250 == 0:
            print(f"Analyzed {image_index + 1}/{len(grouped)} images")

    class_summaries: List[dict] = []
    for class_id, class_name in enumerate(VOC_CLASSES):
        row = confusion[class_id]
        gt_pixels = int(row.sum())
        tp = int(row[class_id])
        fn = gt_pixels - tp
        pred_pixels = int(confusion[:, class_id].sum())
        fp = pred_pixels - tp
        union = tp + fn + fp
        confusions = []
        for predicted_id, pixel_count in enumerate(row):
            if predicted_id == class_id or pixel_count == 0:
                continue
            predicted_name = (
                VOC_CLASSES[predicted_id]
                if predicted_id < len(VOC_CLASSES)
                else "uncovered"
            )
            confusions.append(
                {
                    "predicted_class_id": predicted_id,
                    "predicted_class": predicted_name,
                    "pixels": int(pixel_count),
                    "fraction_of_gt_class": safe_ratio(pixel_count, gt_pixels),
                }
            )
        confusions.sort(key=lambda item: item["pixels"], reverse=True)
        class_summaries.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_pixels": gt_pixels,
                "pred_pixels": pred_pixels,
                "true_positive_pixels": tp,
                "false_negative_pixels": fn,
                "false_positive_pixels": fp,
                "misclassification_rate": safe_ratio(fn, gt_pixels),
                "class_accuracy": safe_ratio(tp, gt_pixels),
                "iou": safe_ratio(tp, union),
                "top_confusions": confusions[:5],
            }
        )

    class_summaries.sort(
        key=lambda item: (
            item["false_negative_pixels"],
            item["misclassification_rate"],
        ),
        reverse=True,
    )
    selected_summaries = class_summaries[: int(args.top_classes)]

    for rank, summary in enumerate(selected_summaries, start=1):
        class_id = int(summary["class_id"])
        candidates = sorted(
            image_metrics[class_id],
            key=lambda item: (
                -item["false_negative_pixels"],
                item["iou"],
                -item["gt_pixels"],
            ),
        )
        selected = candidates[: int(args.images_per_class)]
        summary["misclassification_rank"] = rank
        summary["ranking_metric"] = "false_negative_pixels"
        summary["selected_samples"] = selected
        summary["selection_rule"] = (
            "Highest false-negative pixels for the ground-truth class; "
            "ties prefer lower image-level class IoU and then larger objects."
        )
        summary["feature_visualizations"] = []
        report_name = f"{rank:02d}_{summary['class_name']}.json"
        summary["json_report"] = f"reports/{report_name}"
        write_json(reports_dir / report_name, summary)

    global_summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset": "PASCAL VOC 2012 validation (20 foreground classes)",
        "dataset_root": str(dataset_root),
        "prediction_json": str(prediction_json),
        "evaluated_images": len(grouped),
        "valid_pixels": int(confusion.sum()),
        "uncovered_valid_prediction_pixels": int(uncovered_valid_pixels),
        "ranking_metric": "false_negative_pixels",
        "top_classes_requested": int(args.top_classes),
        "images_per_class": int(args.images_per_class),
        "selected_classes": selected_summaries,
        "all_classes_by_false_negative_pixels": class_summaries,
        "confusion_matrix_rows_gt_columns_pred": confusion.tolist(),
        "class_names": VOC_CLASSES,
        "prediction_sentinel_column": "uncovered",
        "model_eval_metrics_from_prediction_artifact": {
            "mean_iou": float(
                np.nanmean([item["iou"] for item in class_summaries]) * 100.0
            ),
            "mean_class_accuracy": float(
                np.nanmean(
                    [item["class_accuracy"] for item in class_summaries]
                )
                * 100.0
            ),
            "pixel_accuracy": float(
                sum(item["true_positive_pixels"] for item in class_summaries)
                / max(1, int(confusion.sum()))
                * 100.0
            ),
        },
    }
    write_json(output_dir / "summary.json", global_summary)
    write_json(
        output_dir / "confusion_matrix.json",
        {
            "rows": "ground_truth",
            "columns": "prediction",
            "class_names": VOC_CLASSES,
            "prediction_sentinel_column": "uncovered",
            "matrix": confusion.tolist(),
        },
    )
    write_markdown_report(output_dir)
    print(f"Analysis written to {output_dir}")
    return global_summary


def array_stats(array: np.ndarray) -> dict:
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p02": float(np.percentile(finite, 2)),
        "p98": float(np.percentile(finite, 98)),
    }


def pair_stats(with_attr: np.ndarray, without_attr: np.ndarray) -> dict:
    attr_flat = with_attr.astype(np.float64).reshape(-1)
    no_attr_flat = without_attr.astype(np.float64).reshape(-1)
    mae = float(np.mean(np.abs(attr_flat - no_attr_flat)))
    denom = float(np.linalg.norm(attr_flat) * np.linalg.norm(no_attr_flat))
    cosine = (
        float(np.dot(attr_flat, no_attr_flat) / denom)
        if denom > 0
        else None
    )
    if attr_flat.std() > 0 and no_attr_flat.std() > 0:
        correlation = float(np.corrcoef(attr_flat, no_attr_flat)[0, 1])
    else:
        correlation = None
    return {
        "mean_absolute_difference": mae,
        "cosine_similarity": cosine,
        "pearson_correlation": correlation,
    }


def region_contrast(
    feature_map: np.ndarray,
    gt: np.ndarray,
    class_id: int,
) -> dict:
    resized_gt = cv2.resize(
        gt,
        (feature_map.shape[1], feature_map.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    inside = feature_map[resized_gt == class_id]
    outside = feature_map[(resized_gt != class_id) & (resized_gt != 255)]
    inside_mean = float(inside.mean()) if inside.size else None
    outside_mean = float(outside.mean()) if outside.size else None
    return {
        "inside_gt_mean": inside_mean,
        "outside_valid_gt_mean": outside_mean,
        "inside_minus_outside": (
            float(inside_mean - outside_mean)
            if inside_mean is not None and outside_mean is not None
            else None
        ),
    }


def robust_pair_limits(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float]:
    values = np.concatenate(
        [first.astype(np.float64).reshape(-1), second.astype(np.float64).reshape(-1)]
    )
    low, high = np.percentile(values, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def add_title(image_rgb: np.ndarray, title: str) -> np.ndarray:
    canvas = image_rgb.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def feature_overlay(
    image_rgb: np.ndarray,
    feature_map: np.ndarray,
    title: str,
    limits: tuple[float, float],
    alpha: float = 0.48,
) -> np.ndarray:
    low, high = limits
    normalized = np.clip((feature_map - low) / (high - low), 0.0, 1.0)
    heat_bgr = cv2.applyColorMap(
        np.round(normalized * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    heat_rgb = cv2.resize(
        heat_rgb,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, heat_rgb, alpha, 0.0)
    return add_title(blended, f"{title}  [{low:.3f}, {high:.3f}]")


def segmentation_overlay(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    title: str,
    alpha: float = 0.48,
) -> np.ndarray:
    color = np.zeros_like(image_rgb)
    valid = labels < len(VOC_CLASSES)
    color[valid] = VOC_COLORS[labels[valid]]
    blended = image_rgb.copy()
    blended[valid] = np.round(
        (1.0 - alpha) * image_rgb[valid] + alpha * color[valid]
    ).astype(np.uint8)
    return add_title(blended, title)


def target_error_overlay(
    image_rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    class_id: int,
) -> np.ndarray:
    target = gt == class_id
    true_positive = target & (pred == class_id)
    false_negative = target & (pred != class_id)
    colors = np.zeros_like(image_rgb)
    colors[true_positive] = np.asarray([40, 210, 70], dtype=np.uint8)
    colors[false_negative] = np.asarray([240, 45, 45], dtype=np.uint8)
    result = image_rgb.copy()
    visible = true_positive | false_negative
    result[visible] = np.round(
        0.45 * image_rgb[visible] + 0.55 * colors[visible]
    ).astype(np.uint8)
    return add_title(result, "Target: TP green / FN red")


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def build_comparison_grid(
    panels: Sequence[np.ndarray],
    output_path: Path,
) -> None:
    panel_width, panel_height = 420, 300
    fitted = [
        fit_panel(panel, panel_width, panel_height)
        for panel in panels
    ]
    blank = np.full_like(fitted[0], 245)
    while len(fitted) < 8:
        fitted.append(blank.copy())
    row1 = np.concatenate(fitted[:4], axis=1)
    row2 = np.concatenate(fitted[4:8], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path, quality=95)


class FeatureCapture:
    def __init__(self, aggregator: Any) -> None:
        self.aggregator = aggregator
        self.cache: Dict[str, Any] = {}
        self._handles = [
            aggregator.conv1.register_forward_pre_hook(self._capture_cost),
            aggregator.layers[-1].register_forward_hook(self._capture_aggregated),
        ]
        if aggregator.attribute_adapter is not None:
            self._handles.append(
                aggregator.attribute_adapter.register_forward_hook(
                    self._capture_attributes
                )
            )

    def clear(self) -> None:
        self.cache.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def _capture_cost(self, module: Any, inputs: tuple[Any, ...]) -> None:
        import torch

        tensor = inputs[0].detach().float().cpu()
        if tensor.shape[0] % len(VOC_CLASSES) != 0:
            raise ValueError(f"Unexpected cost-volume shape: {tuple(tensor.shape)}")
        batch = tensor.shape[0] // len(VOC_CLASSES)
        self.cache["cost_volume"] = tensor.reshape(
            batch,
            len(VOC_CLASSES),
            tensor.shape[1],
            tensor.shape[2],
            tensor.shape[3],
        ).numpy()

    def _capture_aggregated(
        self,
        module: Any,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        self.cache["aggregated_features"] = (
            output.detach().float().cpu().numpy()
        )

    def _capture_attributes(
        self,
        module: Any,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        self.cache["attribute_retrieval"] = {
            "confidence": output.attr_confidence.detach().float().cpu().numpy(),
            "selected_indices": (
                output.selected_indices.detach().cpu().numpy()
            ),
            "selected_scores": (
                output.selected_scores.detach().float().cpu().numpy()
            ),
            "selected_mask": output.selected_mask.detach().cpu().numpy(),
        }


def build_predictor(args: argparse.Namespace) -> Any:
    os.environ["DETECTRON2_DATASETS"] = str(Path(args.dataset_root).resolve())
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config

    # Import registers CATSeg and its datasets/components.
    from cat_seg import add_cat_seg_config  # noqa: F401

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(str(Path(args.config).resolve()))
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(Path(args.checkpoint).resolve())
    cfg.MODEL.SEM_SEG_HEAD.CACHE_DIR = str(
        Path(args.openclip_pretrained).resolve()
    )
    cfg.MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH = str(
        Path(args.attribute_database).resolve()
    )
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = str(
        Path(args.class_json).resolve()
    )
    cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES = [1, 1]
    # Feature visualization uses a single, whole-image forward. The ranking and
    # masks still come from the supplied sliding-window evaluation artifact.
    cfg.TEST.SLIDING_WINDOW = False
    cfg.DATASETS.TEST = ("voc_2012_test_sem_seg",)
    cfg.freeze()
    torch.set_float32_matmul_precision("high")
    return DefaultPredictor(cfg)


def extract_class_maps(
    cache: Mapping[str, Any],
    class_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    cost = cache["cost_volume"][0, class_id].mean(axis=0)
    aggregated = cache["aggregated_features"][0, :, class_id]
    aggregated_rms = np.sqrt(np.mean(np.square(aggregated), axis=0))
    return cost.astype(np.float32), aggregated_rms.astype(np.float32)


def serialize_retrieval(
    cache: Mapping[str, Any],
    class_id: int,
    representative_phrases: Sequence[str] | None = None,
) -> dict | None:
    retrieval = cache.get("attribute_retrieval")
    if retrieval is None:
        return None
    mask = retrieval["selected_mask"][0, class_id].astype(bool)
    indices = (
        retrieval["selected_indices"][0, class_id][mask].astype(int).tolist()
    )
    payload = {
        "confidence": float(retrieval["confidence"][0, class_id]),
        "selected_cluster_indices": indices,
        "selected_scores": (
            retrieval["selected_scores"][0, class_id][mask]
            .astype(float)
            .tolist()
        ),
    }
    if representative_phrases is not None:
        payload["representative_phrases"] = [
            representative_phrases[index] for index in indices
        ]
    return payload


def visualize_selected(args: argparse.Namespace) -> None:
    import torch

    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    grouped = group_predictions(Path(args.predictions).resolve())
    records_by_stem = {
        Path(file_name).stem: records for file_name, records in grouped.items()
    }

    predictor = build_predictor(args)
    model = predictor.model
    aggregator = model.sem_seg_head.predictor.transformer
    capture = FeatureCapture(aggregator)
    attribute_payload = torch.load(
        str(Path(args.attribute_database).resolve()),
        map_location="cpu",
    )
    representative_phrases = (
        attribute_payload.get("representative_phrases")
        if isinstance(attribute_payload, dict)
        else None
    )
    database_meta = (
        attribute_payload.get("meta", {})
        if isinstance(attribute_payload, dict)
        else {}
    )
    attr_layer = aggregator.layers[aggregator.attr_spatial_layer_index]
    inference_metadata = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "openclip_pretrained": str(Path(args.openclip_pretrained).resolve()),
        "attribute_database": str(Path(args.attribute_database).resolve()),
        "class_json": str(Path(args.class_json).resolve()),
        "device": str(model.device),
        "torch_version": torch.__version__,
        "learned_attribute_gains": {
            "text_attr_gain_raw": float(
                aggregator.attribute_adapter.text_attr_gain.detach().cpu()
            ),
            "text_attr_gain_tanh": float(
                torch.tanh(
                    aggregator.attribute_adapter.text_attr_gain.detach()
                ).cpu()
            ),
            "spatial_block_first_gain_raw": float(
                attr_layer.swin_block_first.attr_spatial_gate.visual_attr_gain
                .detach()
                .cpu()
            ),
            "spatial_block_second_gain_raw": float(
                attr_layer.swin_block_second.attr_spatial_gate.visual_attr_gain
                .detach()
                .cpu()
            ),
        },
        "attribute_database_meta": database_meta,
        "feature_extraction": {
            "cost_volume": (
                "Raw cosine-similarity cost volume immediately before conv1; "
                "prompt channels averaged."
            ),
            "aggregated": (
                "Output of the final AggregatorLayer; channel-wise RMS."
            ),
            "ablation": (
                "Same loaded model/checkpoint; "
                "Aggregator.use_attribute_fusion toggled at inference."
            ),
            "input_mode": (
                "Whole-image Detectron2 DefaultPredictor forward with "
                "TEST.SLIDING_WINDOW=False. Sample ranking and stored prediction "
                "overlay come from the supplied sliding-window evaluation JSON."
            ),
        },
    }
    summary["feature_inference"] = inference_metadata

    try:
        for class_summary in summary["selected_classes"]:
            rank = int(class_summary["misclassification_rank"])
            class_id = int(class_summary["class_id"])
            class_name = class_summary["class_name"]
            class_dir_name = f"{rank:02d}_{class_name}"
            class_dir = output_dir / "classes" / class_dir_name
            class_dir.mkdir(parents=True, exist_ok=True)
            visual_entries: List[dict] = []

            for sample_index, sample in enumerate(
                class_summary["selected_samples"],
                start=1,
            ):
                image_id = sample["image_id"]
                image_bgr = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise FileNotFoundError(sample["image_path"])
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                gt = np.asarray(
                    Image.open(sample["ground_truth_path"]),
                    dtype=np.uint8,
                )
                pred = decode_prediction(records_by_stem[image_id], gt.shape)

                capture.clear()
                aggregator.use_attribute_fusion = True
                with torch.no_grad():
                    predictor(image_bgr)
                attr_cache = dict(capture.cache)
                attr_cost, attr_aggregated = extract_class_maps(
                    attr_cache,
                    class_id,
                )
                retrieval = serialize_retrieval(
                    attr_cache,
                    class_id,
                    representative_phrases=representative_phrases,
                )

                capture.clear()
                aggregator.use_attribute_fusion = False
                with torch.no_grad():
                    predictor(image_bgr)
                no_attr_cache = dict(capture.cache)
                no_attr_cost, no_attr_aggregated = extract_class_maps(
                    no_attr_cache,
                    class_id,
                )

                cost_limits = robust_pair_limits(attr_cost, no_attr_cost)
                aggregated_limits = robust_pair_limits(
                    attr_aggregated,
                    no_attr_aggregated,
                )
                prefix = f"{sample_index:02d}_{image_id}"
                artifact_paths: Dict[str, str] = {}
                maps_and_titles = [
                    (
                        "attr_cost_volume",
                        attr_cost,
                        "Attribute fusion: cost volume",
                        cost_limits,
                    ),
                    (
                        "attr_aggregated",
                        attr_aggregated,
                        "Attribute fusion: aggregated RMS",
                        aggregated_limits,
                    ),
                    (
                        "no_attr_cost_volume",
                        no_attr_cost,
                        "No attribute fusion: cost volume",
                        cost_limits,
                    ),
                    (
                        "no_attr_aggregated",
                        no_attr_aggregated,
                        "No attribute fusion: aggregated RMS",
                        aggregated_limits,
                    ),
                ]
                feature_panels = []
                for key, array, title, limits in maps_and_titles:
                    overlay = feature_overlay(
                        image_rgb,
                        array,
                        title,
                        limits,
                    )
                    image_path = class_dir / f"{prefix}_{key}.png"
                    Image.fromarray(overlay).save(image_path)
                    artifact_paths[key] = str(image_path.relative_to(output_dir))
                    feature_panels.append(overlay)

                original_panel = add_title(image_rgb, "Original image")
                gt_panel = segmentation_overlay(
                    image_rgb,
                    gt,
                    f"Ground truth: {class_name}",
                )
                pred_panel = segmentation_overlay(
                    image_rgb,
                    pred,
                    "Stored evaluation prediction",
                )
                error_panel = target_error_overlay(
                    image_rgb,
                    gt,
                    pred,
                    class_id,
                )
                comparison_path = class_dir / f"{prefix}_comparison.png"
                build_comparison_grid(
                    [
                        original_panel,
                        gt_panel,
                        pred_panel,
                        error_panel,
                        *feature_panels,
                    ],
                    comparison_path,
                )
                artifact_paths["comparison"] = str(
                    comparison_path.relative_to(output_dir)
                )
                npz_path = class_dir / f"{prefix}_feature_maps.npz"
                np.savez_compressed(
                    npz_path,
                    attr_cost_volume=attr_cost,
                    attr_aggregated_rms=attr_aggregated,
                    no_attr_cost_volume=no_attr_cost,
                    no_attr_aggregated_rms=no_attr_aggregated,
                )
                artifact_paths["raw_feature_maps"] = str(
                    npz_path.relative_to(output_dir)
                )

                feature_metrics = {
                    "cost_volume": {
                        "with_attribute_fusion": {
                            **array_stats(attr_cost),
                            "gt_region_contrast": region_contrast(
                                attr_cost,
                                gt,
                                class_id,
                            ),
                        },
                        "without_attribute_fusion": {
                            **array_stats(no_attr_cost),
                            "gt_region_contrast": region_contrast(
                                no_attr_cost,
                                gt,
                                class_id,
                            ),
                        },
                        "paired_comparison": pair_stats(
                            attr_cost,
                            no_attr_cost,
                        ),
                    },
                    "aggregated_features": {
                        "reduction": "channel_rms",
                        "with_attribute_fusion": {
                            **array_stats(attr_aggregated),
                            "gt_region_contrast": region_contrast(
                                attr_aggregated,
                                gt,
                                class_id,
                            ),
                        },
                        "without_attribute_fusion": {
                            **array_stats(no_attr_aggregated),
                            "gt_region_contrast": region_contrast(
                                no_attr_aggregated,
                                gt,
                                class_id,
                            ),
                        },
                        "paired_comparison": pair_stats(
                            attr_aggregated,
                            no_attr_aggregated,
                        ),
                    },
                }
                visual_entries.append(
                    {
                        "image_id": image_id,
                        "sample_rank_within_class": sample_index,
                        "stored_prediction_metrics": {
                            key: sample[key]
                            for key in (
                                "gt_pixels",
                                "pred_pixels",
                                "true_positive_pixels",
                                "false_negative_pixels",
                                "false_positive_pixels",
                                "class_accuracy",
                                "iou",
                            )
                        },
                        "attribute_retrieval": retrieval,
                        "feature_metrics": feature_metrics,
                        "artifacts": artifact_paths,
                    }
                )
                print(
                    f"Visualized class {rank}/{len(summary['selected_classes'])} "
                    f"{class_name}: sample {sample_index}/"
                    f"{len(class_summary['selected_samples'])} ({image_id})"
                )

            class_summary["feature_visualizations"] = visual_entries
            summarize_class_features(class_summary)
            report_path = output_dir / class_summary["json_report"]
            write_json(report_path, class_summary)
    finally:
        aggregator.use_attribute_fusion = True
        capture.close()

    summary["visualization_completed_at"] = (
        datetime.now().astimezone().isoformat()
    )
    write_json(summary_path, summary)
    write_markdown_report(output_dir)
    print(f"Feature visualization written to {output_dir}")


def percent(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number * 100.0:.2f}%"


def finite_mean(values: Iterable[Any]) -> float | None:
    finite_values = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            finite_values.append(number)
    return (
        float(sum(finite_values) / len(finite_values))
        if finite_values
        else None
    )


def summarize_class_features(class_summary: MutableMapping[str, Any]) -> dict:
    entries = class_summary.get("feature_visualizations", [])
    cost_mae = [
        entry["feature_metrics"]["cost_volume"]["paired_comparison"][
            "mean_absolute_difference"
        ]
        for entry in entries
    ]
    aggregate_mae = [
        entry["feature_metrics"]["aggregated_features"]["paired_comparison"][
            "mean_absolute_difference"
        ]
        for entry in entries
    ]
    cost_normalized = []
    aggregate_normalized = []
    cost_contrast_delta = []
    aggregate_contrast_delta = []
    confidences = []
    phrases: List[str] = []
    for entry in entries:
        cost = entry["feature_metrics"]["cost_volume"]
        aggregate = entry["feature_metrics"]["aggregated_features"]
        cost_std = cost["without_attribute_fusion"].get("std")
        aggregate_std = aggregate["without_attribute_fusion"].get("std")
        cost_difference = cost["paired_comparison"]["mean_absolute_difference"]
        aggregate_difference = aggregate["paired_comparison"][
            "mean_absolute_difference"
        ]
        if cost_std:
            cost_normalized.append(cost_difference / cost_std)
        if aggregate_std:
            aggregate_normalized.append(aggregate_difference / aggregate_std)

        for stage, destination in (
            (cost, cost_contrast_delta),
            (aggregate, aggregate_contrast_delta),
        ):
            with_attr = stage["with_attribute_fusion"][
                "gt_region_contrast"
            ]["inside_minus_outside"]
            without_attr = stage["without_attribute_fusion"][
                "gt_region_contrast"
            ]["inside_minus_outside"]
            if with_attr is not None and without_attr is not None:
                destination.append(float(with_attr - without_attr))

        retrieval = entry.get("attribute_retrieval") or {}
        confidence = retrieval.get("confidence")
        if confidence is not None:
            confidences.append(confidence)
        for phrase in retrieval.get("representative_phrases", []):
            if phrase not in phrases:
                phrases.append(phrase)

    top_confusion = (
        class_summary["top_confusions"][0]
        if class_summary.get("top_confusions")
        else None
    )
    result = {
        "sample_count": len(entries),
        "mean_attribute_confidence": finite_mean(confidences),
        "mean_cost_volume_mae": finite_mean(cost_mae),
        "mean_aggregated_feature_mae": finite_mean(aggregate_mae),
        "mean_cost_volume_mae_over_no_attr_std": finite_mean(
            cost_normalized
        ),
        "mean_aggregated_mae_over_no_attr_std": finite_mean(
            aggregate_normalized
        ),
        "mean_attribute_delta_in_gt_region_contrast": {
            "cost_volume": finite_mean(cost_contrast_delta),
            "aggregated_features": finite_mean(aggregate_contrast_delta),
            "available_cost_samples": len(cost_contrast_delta),
            "available_aggregated_samples": len(aggregate_contrast_delta),
        },
        "representative_retrieved_phrases": phrases[:12],
        "dominant_confusion": top_confusion,
        "interpretation": [],
    }
    if entries:
        result["interpretation"].append(
            "属性开/关在所有代表图上均产生非零差异；平均标准化 MAE "
            f"为 cost-volume {result['mean_cost_volume_mae_over_no_attr_std']:.4f}、"
            "聚合特征 "
            f"{result['mean_aggregated_mae_over_no_attr_std']:.4f}。"
        )
    contrast = result["mean_attribute_delta_in_gt_region_contrast"]
    if contrast["available_cost_samples"]:
        direction = (
            "增强"
            if float(contrast["cost_volume"]) > 0
            else "减弱"
        )
        result["interpretation"].append(
            f"在可计算目标区内外对比的样本中，属性融合平均{direction}"
            "了 cost-volume 的目标区分离度 "
            f"{abs(float(contrast['cost_volume'])):.6f}。"
        )
    if top_confusion is not None:
        result["interpretation"].append(
            f"该类最主要被混淆为 {top_confusion['predicted_class']}，"
            f"涉及 {top_confusion['pixels']} 个 GT 像素。"
        )
    class_summary["feature_summary"] = result
    return result


def write_markdown_report(output_dir: Path) -> None:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    metrics = summary["model_eval_metrics_from_prediction_artifact"]
    lines = [
        "# PASCAL VOC 预测错误与属性融合特征可视化分析",
        "",
        f"- 数据集：{summary['dataset']}",
        f"- 测试图片数：{summary['evaluated_images']}",
        f"- 预测来源：`{summary['prediction_json']}`",
        (
            "- 误分类排序口径：按真实类别像素中被预测成其他类别的"
            "像素数（false-negative pixels）降序"
        ),
        (
            f"- 由预测 JSON 复算：mIoU {metrics['mean_iou']:.4f}%，"
            f"mACC {metrics['mean_class_accuracy']:.4f}%，"
            f"pACC {metrics['pixel_accuracy']:.4f}%"
        ),
        "",
        "## 误分类最多的类别",
        "",
        "| 排名 | 类别 | FN 像素 | 错误率 | IoU | 主要混淆类别 | JSON |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for item in summary["selected_classes"]:
        top_confusions = ", ".join(
            f"{entry['predicted_class']} ({entry['pixels']})"
            for entry in item["top_confusions"][:3]
        )
        lines.append(
            "| {rank} | {name} | {fn:,} | {error} | {iou} | {conf} | "
            "[报告]({json_path}) |".format(
                rank=item["misclassification_rank"],
                name=item["class_name"],
                fn=item["false_negative_pixels"],
                error=percent(item["misclassification_rate"]),
                iou=percent(item["iou"]),
                conf=top_confusions or "无",
                json_path=item["json_report"],
            )
        )

    lines.extend(
        [
            "",
            "## 可视化方法",
            "",
            (
                "每张代表图均给出四种原图叠加结果：属性融合后的 "
                "cost-volume、属性融合后的末层聚合特征、不使用属性融合的 "
                "cost-volume，以及不使用属性融合的末层聚合特征。"
            ),
            (
                "cost-volume 取进入 `conv1` 前目标类别的原始余弦相似度"
                "（多 prompt 时取均值）；聚合特征取最后一个 "
                "`AggregatorLayer` 输出在通道维的 RMS。"
            ),
            (
                "属性开/关使用同一个已加载 checkpoint 和同一模型实例，"
                "只切换 `Aggregator.use_attribute_fusion`，因此非属性分支"
                "权重完全一致。每一对开/关热图共享 2%–98% 分位颜色范围。"
            ),
            "",
        ]
    )
    if "feature_inference" not in summary:
        lines.extend(
            [
                "> 特征推理尚未执行；当前报告只包含预测错误统计与样本选择。",
                "",
            ]
        )

    for item in summary["selected_classes"]:
        feature_summary = item.get("feature_summary")
        lines.extend(
            [
                f"## {item['misclassification_rank']}. {item['class_name']}",
                "",
                (
                    f"全数据集：GT {item['gt_pixels']:,} 像素，"
                    f"FN {item['false_negative_pixels']:,}，"
                    f"FP {item['false_positive_pixels']:,}，"
                    f"IoU {percent(item['iou'])}。"
                ),
                "",
            ]
        )
        if feature_summary:
            contrast = feature_summary[
                "mean_attribute_delta_in_gt_region_contrast"
            ]
            lines.extend(
                [
                    (
                        "属性开/关平均差异：cost-volume MAE "
                        f"{feature_summary['mean_cost_volume_mae']:.6f}"
                        "（相对 no-attr 标准差 "
                        f"{feature_summary['mean_cost_volume_mae_over_no_attr_std']:.4f}），"
                        "聚合特征 MAE "
                        f"{feature_summary['mean_aggregated_feature_mae']:.6f}"
                        "（相对 no-attr 标准差 "
                        f"{feature_summary['mean_aggregated_mae_over_no_attr_std']:.4f}）。"
                    ),
                    "",
                    (
                        "平均属性召回置信度："
                        f"{feature_summary['mean_attribute_confidence']:.4f}；"
                        "代表性召回短语："
                        + "；".join(
                            feature_summary[
                                "representative_retrieved_phrases"
                            ][:5]
                        )
                        + "。"
                    ),
                    "",
                ]
            )
            if contrast["available_cost_samples"]:
                lines.extend(
                    [
                        (
                            "目标 GT 区域相对非目标有效区域的激活对比，"
                            "属性融合带来的平均变化为：cost-volume "
                            f"{contrast['cost_volume']:+.6f}，聚合特征 "
                            f"{contrast['aggregated_features']:+.6f}"
                            f"（可计算样本 {contrast['available_cost_samples']} 张；"
                            "其余样本的有效 GT 像素只有该目标类，无法定义区外对比）。"
                        ),
                        "",
                    ]
                )
        visualization_by_id = {
            entry["image_id"]: entry
            for entry in item.get("feature_visualizations", [])
        }
        for sample_rank, sample in enumerate(
            item["selected_samples"],
            start=1,
        ):
            lines.extend(
                [
                    f"### 样本 {sample_rank}：{sample['image_id']}",
                    "",
                    (
                        f"目标类别 GT {sample['gt_pixels']:,} 像素，"
                        f"FN {sample['false_negative_pixels']:,}，"
                        f"FP {sample['false_positive_pixels']:,}，"
                        f"图像级类别 IoU {percent(sample['iou'])}。"
                    ),
                    "",
                ]
            )
            visual = visualization_by_id.get(sample["image_id"])
            if visual:
                comparison = visual["artifacts"]["comparison"]
                cost_compare = visual["feature_metrics"]["cost_volume"][
                    "paired_comparison"
                ]
                aggregate_compare = visual["feature_metrics"][
                    "aggregated_features"
                ]["paired_comparison"]
                lines.extend(
                    [
                        f"![{item['class_name']} {sample['image_id']}]"
                        f"({comparison})",
                        "",
                        (
                            "属性开/关差异：cost-volume MAE "
                            f"{cost_compare['mean_absolute_difference']:.6f}，"
                            "聚合特征 MAE "
                            f"{aggregate_compare['mean_absolute_difference']:.6f}。"
                        ),
                        "",
                    ]
                )

    lines.extend(
        [
            "## 解释边界",
            "",
            (
                "类别与代表图的排名来自给定的滑窗评估预测 JSON；特征图为"
                "同一图片的整图前向，以避免将多个滑窗 patch 的内部特征做"
                "未经训练定义的拼接。两者的用途分别是错误筛选与机制观察。"
            ),
            (
                "热图展示激活位置与相对强弱，不等价于因果归因。更可靠的"
                "结论应结合逐图 IoU、GT 区域内外激活对比及属性开/关差值。"
            ),
            "",
            "## 文件说明",
            "",
            "- `summary.json`：完整类别排序、混淆矩阵、运行元数据。",
            "- `confusion_matrix.json`：20×21 的 GT→预测像素混淆矩阵。",
            "- `reports/*.json`：每个入选类别的独立 JSON 报告。",
            "- `classes/*/*_comparison.png`：原图、GT、预测及四组特征图总览。",
            "- `classes/*/*_feature_maps.npz`：四组原始低分辨率特征数组。",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def finalize_existing_reports(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    summary_path = output_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    for class_summary in summary["selected_classes"]:
        summarize_class_features(class_summary)
        write_json(output_dir / class_summary["json_report"], class_summary)
    write_json(summary_path, summary)
    write_markdown_report(output_dir)
    print(f"Reports finalized in {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("analyze", "visualize", "report", "all"),
        default="all",
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-classes", type=int, default=5)
    parser.add_argument("--images-per-class", type=int, default=3)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/eva_vitb_384_attr_fusion.yaml"),
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--openclip-pretrained")
    parser.add_argument("--attribute-database")
    parser.add_argument(
        "--class-json",
        default=str(PROJECT_ROOT / "datasets/voc20.json"),
    )
    return parser


def validate_visualization_args(args: argparse.Namespace) -> None:
    required = (
        "checkpoint",
        "openclip_pretrained",
        "attribute_database",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError(
            "Visualization requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.phase in ("analyze", "all"):
        analyze_predictions(args)
    if args.phase in ("visualize", "all"):
        validate_visualization_args(args)
        visualize_selected(args)
    elif args.phase == "report":
        finalize_existing_reports(Path(args.output_dir))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualize PASCAL VOC-20 confusion pairs with attribute-enhanced features.

The script consumes the prediction artifact produced by sliding-window
evaluation, ranks GT classes by off-diagonal confused pixels, and selects the
worst validation image for the dominant wrong class of each selected GT class.
It then reruns exactly the configured 4-window + 1-global-view inference and
captures:

* the attribute-enhanced cost volume before ``Aggregator.conv1``;
* the final per-class segmentation logits returned by ``Aggregator``;
* image-conditioned attribute retrieval details and feature vectors.

Low-resolution view features are stitched with the same 384/256 geometry and
global-view averaging used by ``CATSeg.inference_sliding_window``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualize_voc_feature_ablation import (  # noqa: E402
    VOC_CLASSES,
    add_title,
    decode_prediction,
    feature_overlay,
    group_predictions,
    robust_pair_limits,
    segmentation_overlay,
    write_json,
)


VIEW_NAMES = ("window_top_left", "window_top_right", "window_bottom_left", "window_bottom_right", "global")
WINDOW_COORDS = ((0, 0), (0, 256), (256, 0), (256, 256))


def build_predictor(args: argparse.Namespace) -> Any:
    """Build the exact attribute-enhanced sliding-window validation model."""
    os.environ["DETECTRON2_DATASETS"] = str(Path(args.dataset_root).resolve())
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config

    from cat_seg import add_cat_seg_config

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
    cfg.TEST.SLIDING_WINDOW = True
    cfg.DATASETS.TEST = ("voc_2012_test_sem_seg",)
    cfg.freeze()
    torch.set_float32_matmul_precision("high")
    return DefaultPredictor(cfg)


class SlidingFeatureCapture:
    """Capture class-wise tensors for all four windows and the global view."""

    def __init__(self, aggregator: Any) -> None:
        self.cache: Dict[str, Any] = {}
        self._handles = [
            aggregator.conv1.register_forward_pre_hook(self._capture_cost),
            aggregator.register_forward_hook(self._capture_logits),
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
        tensor = inputs[0].detach().float().cpu()
        class_count = len(VOC_CLASSES)
        if tensor.shape[0] % class_count:
            raise ValueError(f"Unexpected cost shape: {tuple(tensor.shape)}")
        batch = tensor.shape[0] // class_count
        self.cache["cost"] = tensor.reshape(
            batch,
            class_count,
            tensor.shape[1],
            tensor.shape[2],
            tensor.shape[3],
        ).numpy()

    def _capture_logits(
        self,
        module: Any,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        tensor = output.detach().float().cpu()
        if tensor.ndim != 4 or tensor.shape[1] != len(VOC_CLASSES):
            raise ValueError(f"Unexpected final-logit shape: {tuple(tensor.shape)}")
        self.cache["final_logits"] = tensor.numpy()

    def _capture_attributes(
        self,
        module: Any,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        self.cache["retrieval"] = {
            "confidence": output.attr_confidence.detach().float().cpu().numpy(),
            "indices": output.selected_indices.detach().cpu().numpy(),
            "scores": output.selected_scores.detach().float().cpu().numpy(),
            "mask": output.selected_mask.detach().cpu().numpy(),
        }


def stitch_sliding_views(view_maps: np.ndarray) -> np.ndarray:
    """Reconstruct a 640x640 map from four local views and one global view."""
    if view_maps.shape[0] != 5:
        raise ValueError(f"Expected 5 sliding views, got {view_maps.shape[0]}")
    accum = np.zeros((640, 640), dtype=np.float32)
    counts = np.zeros((640, 640), dtype=np.float32)
    for view, (top, left) in zip(view_maps[:4], WINDOW_COORDS):
        resized = cv2.resize(
            view.astype(np.float32),
            (384, 384),
            interpolation=cv2.INTER_LINEAR,
        )
        accum[top : top + 384, left : left + 384] += resized
        counts[top : top + 384, left : left + 384] += 1.0
    local = accum / np.maximum(counts, 1.0)
    global_map = cv2.resize(
        view_maps[4].astype(np.float32),
        (640, 640),
        interpolation=cv2.INTER_LINEAR,
    )
    return (local + global_map) / 2.0


def reconstruct_pre_postprocess_mask_score(
    logit_views: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the mask score immediately before sem_seg_postprocess.

    This follows ``CATSeg.inference_sliding_window`` exactly: interpolate raw
    96x96 logits to 384x384, apply sigmoid, fold the four local windows, resize
    the global probability view to 640x640, and average local/global scores.
    """
    import torch
    from torch.nn import functional as F

    tensor = torch.from_numpy(logit_views.astype(np.float32))[:, None]
    resized_logits = F.interpolate(
        tensor,
        size=(384, 384),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    mask_scores = resized_logits.sigmoid()
    local_scores = mask_scores[:4].numpy()
    accum = np.zeros((640, 640), dtype=np.float32)
    counts = np.zeros((640, 640), dtype=np.float32)
    for view, (top, left) in zip(local_scores, WINDOW_COORDS):
        accum[top : top + 384, left : left + 384] += view
        counts[top : top + 384, left : left + 384] += 1.0
    local = accum / np.maximum(counts, 1.0)
    global_score = F.interpolate(
        mask_scores[4:5, None],
        size=(640, 640),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    stitched = (local + global_score) / 2.0
    return (
        stitched.astype(np.float32),
        resized_logits.numpy().astype(np.float32),
        mask_scores.numpy().astype(np.float32),
    )


def extract_stitched_maps(
    cache: Mapping[str, Any],
    class_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cost_views = cache["cost"][:, class_id].mean(axis=1)
    final_logit_views = cache["final_logits"][:, class_id]
    final_mask_score, resized_logits, mask_score_views = (
        reconstruct_pre_postprocess_mask_score(final_logit_views)
    )
    return (
        stitch_sliding_views(cost_views),
        final_mask_score,
        cost_views.astype(np.float32),
        final_logit_views.astype(np.float32),
        resized_logits,
        mask_score_views,
    )


def pair_error_overlay(
    image_rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    gt_id: int,
    wrong_id: int,
) -> np.ndarray:
    target = gt == gt_id
    correct = target & (pred == gt_id)
    confused = target & (pred == wrong_id)
    other_error = target & (pred != gt_id) & (pred != wrong_id)
    colors = np.zeros_like(image_rgb)
    colors[correct] = [35, 205, 70]
    colors[confused] = [240, 40, 40]
    colors[other_error] = [255, 175, 30]
    visible = correct | confused | other_error
    result = image_rgb.copy()
    result[visible] = np.round(
        0.42 * image_rgb[visible] + 0.58 * colors[visible]
    ).astype(np.uint8)
    return add_title(result, "GT: green=correct red=pair orange=other error")


def fit_panel(image: np.ndarray, width: int = 440, height: int = 320) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def save_grid(panels: Sequence[np.ndarray], path: Path) -> None:
    fitted = [fit_panel(panel) for panel in panels]
    while len(fitted) < 8:
        fitted.append(np.full_like(fitted[0], 245))
    grid = np.concatenate(
        [np.concatenate(fitted[:4], axis=1), np.concatenate(fitted[4:8], axis=1)],
        axis=0,
    )
    Image.fromarray(grid).save(path)


def analyze_predictions(args: argparse.Namespace) -> dict:
    grouped = group_predictions(Path(args.predictions).resolve())
    dataset_root = Path(args.dataset_root).resolve()
    gt_dir = dataset_root / "VOCdevkit/VOC2012/annotations_detectron2/val"
    class_count = len(VOC_CLASSES)
    confusion = np.zeros((class_count, class_count + 1), dtype=np.int64)
    pair_candidates: Dict[tuple[int, int], List[dict]] = defaultdict(list)

    for index, (image_path_string, records) in enumerate(sorted(grouped.items())):
        image_path = Path(image_path_string)
        image_id = image_path.stem
        gt_path = gt_dir / f"{image_id}.png"
        gt = np.asarray(Image.open(gt_path), dtype=np.uint8)
        pred = decode_prediction(records, gt.shape)
        valid = gt != 255
        encoded = (class_count + 1) * gt[valid].astype(np.int64) + pred[valid]
        confusion += np.bincount(
            encoded,
            minlength=class_count * (class_count + 1),
        ).reshape(confusion.shape)

        for gt_value in np.unique(gt[valid]):
            gt_id = int(gt_value)
            gt_pixels = int(np.count_nonzero(gt == gt_id))
            predicted_ids, counts = np.unique(pred[gt == gt_id], return_counts=True)
            for predicted_id_value, count_value in zip(predicted_ids, counts):
                predicted_id = int(predicted_id_value)
                if predicted_id == gt_id or predicted_id >= class_count:
                    continue
                pair_candidates[(gt_id, predicted_id)].append(
                    {
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "ground_truth_path": str(gt_path),
                        "gt_pixels": gt_pixels,
                        "pair_error_pixels": int(count_value),
                        "pair_error_fraction_of_gt": float(count_value / gt_pixels),
                    }
                )
        if (index + 1) % 250 == 0:
            print(f"Analyzed {index + 1}/{len(grouped)} images")

    rankings = []
    for gt_id, name in enumerate(VOC_CLASSES):
        row = confusion[gt_id]
        other = row.copy()
        other[gt_id] = 0
        wrong_id = int(np.argmax(other[:class_count]))
        pair_pixels = int(other[wrong_id])
        fn_pixels = int(other.sum())
        gt_pixels = int(row.sum())
        candidates = sorted(
            pair_candidates[(gt_id, wrong_id)],
            key=lambda item: (
                item["pair_error_pixels"],
                item["pair_error_fraction_of_gt"],
                item["gt_pixels"],
            ),
            reverse=True,
        )
        rankings.append(
            {
                "gt_class_id": gt_id,
                "gt_class": name,
                "wrong_class_id": wrong_id,
                "wrong_class": VOC_CLASSES[wrong_id],
                "gt_pixels": gt_pixels,
                "false_negative_pixels": fn_pixels,
                "misclassification_rate": float(fn_pixels / gt_pixels),
                "dominant_pair_pixels": pair_pixels,
                "dominant_pair_fraction_of_gt": float(pair_pixels / gt_pixels),
                "selected_sample": candidates[0],
            }
        )
    rankings.sort(
        key=lambda item: (
            item["false_negative_pixels"],
            item["misclassification_rate"],
        ),
        reverse=True,
    )
    selected = rankings[: args.top_classes]
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank

    return {
        "dataset": "PASCAL VOC 2012 validation (20 foreground classes)",
        "generated_at": datetime.now().astimezone().isoformat(),
        "evaluated_images": len(grouped),
        "prediction_json": str(Path(args.predictions).resolve()),
        "ranking_rule": "GT classes ranked by total off-diagonal pixels",
        "sample_rule": "For each GT class, choose its dominant wrong class, then the image with the most GT-to-wrong-class pixels",
        "class_names": VOC_CLASSES,
        "confusion_matrix_rows_gt_columns_pred_plus_uncovered": confusion.tolist(),
        "all_class_rankings": rankings,
        "selected_classes": selected,
    }


def heat_color(value: float) -> tuple[int, int, int]:
    pixel = np.asarray([[round(np.clip(value, 0.0, 1.0) * 255)]], dtype=np.uint8)
    bgr = cv2.applyColorMap(pixel, cv2.COLORMAP_VIRIDIS)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def abbreviated_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def render_matrix(
    values: np.ndarray,
    annotations: np.ndarray,
    path: Path,
    title: str,
    annotation_mode: str,
) -> None:
    cell = 58
    left = 190
    top = 260
    bottom = 50
    size = len(VOC_CLASSES)
    canvas = np.full(
        (top + size * cell + bottom, left + size * cell + 30, 3),
        250,
        dtype=np.uint8,
    )
    cv2.putText(canvas, title, (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Rows: ground truth    Columns: prediction", (35, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 60, 60), 1, cv2.LINE_AA)
    for idx, name in enumerate(VOC_CLASSES):
        y = top + idx * cell + cell // 2 + 6
        cv2.putText(canvas, name, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
        # Rotate labels by rendering to a temporary strip.
        strip = np.full((30, 165, 3), 250, dtype=np.uint8)
        cv2.putText(strip, name, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1, cv2.LINE_AA)
        rotated = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)
        y0 = max(5, top - rotated.shape[0] - 4)
        x0 = left + idx * cell + (cell - rotated.shape[1]) // 2
        canvas[y0 : y0 + rotated.shape[0], x0 : x0 + rotated.shape[1]] = rotated

    for row in range(size):
        for col in range(size):
            y0 = top + row * cell
            x0 = left + col * cell
            color = heat_color(float(values[row, col]))
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), color, -1)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), (220, 220, 220), 1)
            if annotation_mode == "percent":
                label = f"{float(annotations[row, col]) * 100:.1f}"
            else:
                label = abbreviated_count(int(annotations[row, col]))
            text_color = (245, 245, 245) if values[row, col] > 0.52 else (20, 20, 20)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)[0]
            cv2.putText(
                canvas,
                label,
                (x0 + (cell - text_size[0]) // 2, y0 + cell // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.37,
                text_color,
                1,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(path), canvas)


def save_confusion_artifacts(summary: Mapping[str, Any], output_dir: Path) -> None:
    confusion = np.asarray(
        summary["confusion_matrix_rows_gt_columns_pred_plus_uncovered"],
        dtype=np.int64,
    )
    matrix = confusion[:, : len(VOC_CLASSES)]
    row_sum = confusion.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_sum, 1)
    log_values = np.log1p(matrix.astype(np.float64))
    log_values /= max(float(log_values.max()), 1.0)
    render_matrix(
        normalized,
        normalized,
        output_dir / "confusion_matrix_row_normalized.png",
        "PASCAL VOC-20 Confusion Matrix (row-normalized, %)",
        "percent",
    )
    render_matrix(
        log_values,
        matrix,
        output_dir / "confusion_matrix_log_counts.png",
        "PASCAL VOC-20 Confusion Matrix (log-scaled pixel counts)",
        "count",
    )
    with (output_dir / "confusion_matrix_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["gt/pred", *VOC_CLASSES, "uncovered"])
        for name, row in zip(VOC_CLASSES, confusion):
            writer.writerow([name, *row.tolist()])


def serialize_retrieval(
    retrieval: Mapping[str, np.ndarray],
    phrases: Sequence[str] | None,
) -> tuple[dict, np.ndarray]:
    indices_array = retrieval["indices"]
    scores_array = retrieval["scores"]
    masks_array = retrieval["mask"].astype(bool)
    confidences = retrieval["confidence"]
    class_payloads = []
    used_indices = set()
    for class_id, class_name in enumerate(VOC_CLASSES):
        views = []
        for view_id, view_name in enumerate(VIEW_NAMES):
            mask = masks_array[view_id, class_id]
            indices = indices_array[view_id, class_id][mask].astype(int)
            scores = scores_array[view_id, class_id][mask].astype(float)
            attributes = []
            for index, score in zip(indices.tolist(), scores.tolist()):
                used_indices.add(index)
                attributes.append(
                    {
                        "cluster_index": index,
                        "score": score,
                        "representative_phrase": (
                            phrases[index] if phrases is not None else None
                        ),
                    }
                )
            views.append(
                {
                    "view": view_name,
                    "confidence": float(confidences[view_id, class_id]),
                    "attributes": attributes,
                }
            )
        class_payloads.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "enhanced": any(view["attributes"] for view in views),
                "mean_confidence": float(confidences[:, class_id].mean()),
                "max_confidence": float(confidences[:, class_id].max()),
                "views": views,
            }
        )
    return {
        "view_names": VIEW_NAMES,
        "enhanced_classes": [
            item["class_name"] for item in class_payloads if item["enhanced"]
        ],
        "classes": class_payloads,
    }, np.asarray(sorted(used_indices), dtype=np.int64)


def visualize_selected(args: argparse.Namespace, summary: dict) -> None:
    import torch

    output_dir = Path(args.output_dir).resolve()
    grouped = group_predictions(Path(args.predictions).resolve())
    records_by_stem = {
        Path(file_name).stem: records for file_name, records in grouped.items()
    }
    predictor = build_predictor(args)
    model = predictor.model
    aggregator = model.sem_seg_head.predictor.transformer
    if not aggregator.use_attribute_fusion:
        raise RuntimeError("Attribute fusion must be enabled for this analysis")
    capture = SlidingFeatureCapture(aggregator)

    database = torch.load(
        str(Path(args.attribute_database).resolve()),
        map_location="cpu",
    )
    phrases = database.get("representative_phrases") if isinstance(database, dict) else None
    embeddings = database.get("embeddings") if isinstance(database, dict) else database
    text_gain = float(torch.tanh(aggregator.attribute_adapter.text_attr_gain.detach()).cpu())
    summary["feature_inference"] = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "openclip_pretrained": str(Path(args.openclip_pretrained).resolve()),
        "attribute_database": str(Path(args.attribute_database).resolve()),
        "class_json": str(Path(args.class_json).resolve()),
        "sliding_window": True,
        "sliding_views": list(VIEW_NAMES),
        "window_kernel": 384,
        "window_stride": 256,
        "stitch_resolution": [640, 640],
        "cost_volume_text": "attribute-enhanced text features",
        "final_mask_score": (
            "Aggregator raw class logits are interpolated to 384x384, passed "
            "through sigmoid, folded across four local windows, averaged with "
            "the global view, and captured immediately before sem_seg_postprocess"
        ),
        "raw_logits_saved": True,
        "text_attribute_gain_tanh": text_gain,
    }

    try:
        for item in summary["selected_classes"]:
            rank = int(item["rank"])
            gt_id = int(item["gt_class_id"])
            wrong_id = int(item["wrong_class_id"])
            gt_name = item["gt_class"]
            wrong_name = item["wrong_class"]
            sample = item["selected_sample"]
            image_id = sample["image_id"]
            class_dir = output_dir / "high_confusion_classes" / f"{rank:02d}_{gt_name}_to_{wrong_name}"
            class_dir.mkdir(parents=True, exist_ok=True)

            image_bgr = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(sample["image_path"])
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            gt = np.asarray(Image.open(sample["ground_truth_path"]), dtype=np.uint8)
            pred = decode_prediction(records_by_stem[image_id], gt.shape)

            capture.clear()
            with torch.no_grad():
                predictor(image_bgr)
            cache = dict(capture.cache)
            (
                gt_cost,
                gt_mask_score,
                gt_cost_views,
                gt_logit_views,
                gt_resized_logits,
                gt_mask_score_views,
            ) = extract_stitched_maps(cache, gt_id)
            (
                wrong_cost,
                wrong_mask_score,
                wrong_cost_views,
                wrong_logit_views,
                wrong_resized_logits,
                wrong_mask_score_views,
            ) = extract_stitched_maps(cache, wrong_id)

            cost_limits = robust_pair_limits(gt_cost, wrong_cost)
            mask_score_limits = robust_pair_limits(
                gt_mask_score,
                wrong_mask_score,
            )
            maps = [
                ("gt_cost_volume", gt_cost, f"GT {gt_name} cost (enhanced text)", cost_limits),
                ("wrong_cost_volume", wrong_cost, f"Wrong {wrong_name} cost (enhanced text)", cost_limits),
                (
                    "gt_final_mask_score",
                    gt_mask_score,
                    f"GT {gt_name} final mask score",
                    mask_score_limits,
                ),
                (
                    "wrong_final_mask_score",
                    wrong_mask_score,
                    f"Wrong {wrong_name} final mask score",
                    mask_score_limits,
                ),
            ]
            feature_panels = []
            artifacts = {}
            for key, array, title, limits in maps:
                overlay = feature_overlay(image_rgb, array, title, limits)
                path = class_dir / f"{image_id}_{key}.png"
                Image.fromarray(overlay).save(path)
                feature_panels.append(overlay)
                artifacts[key] = str(path.relative_to(output_dir))

            original = add_title(image_rgb, f"Original: {image_id}")
            gt_panel = segmentation_overlay(image_rgb, gt, "Ground truth")
            pred_panel = segmentation_overlay(image_rgb, pred, "Stored sliding-window prediction")
            error_panel = pair_error_overlay(image_rgb, gt, pred, gt_id, wrong_id)
            overview = class_dir / f"{image_id}_overview.png"
            save_grid([original, gt_panel, pred_panel, error_panel, *feature_panels], overview)
            artifacts["overview"] = str(overview.relative_to(output_dir))

            npz_path = class_dir / f"{image_id}_activation_maps.npz"
            np.savez_compressed(
                npz_path,
                gt_cost_volume_stitched=gt_cost,
                wrong_cost_volume_stitched=wrong_cost,
                gt_final_mask_score_pre_postprocess=gt_mask_score,
                wrong_final_mask_score_pre_postprocess=wrong_mask_score,
                gt_cost_volume_views=gt_cost_views,
                wrong_cost_volume_views=wrong_cost_views,
                gt_final_logits_views=gt_logit_views,
                wrong_final_logits_views=wrong_logit_views,
                gt_final_logits_resized_384=gt_resized_logits,
                wrong_final_logits_resized_384=wrong_resized_logits,
                gt_mask_score_views_384=gt_mask_score_views,
                wrong_mask_score_views_384=wrong_mask_score_views,
            )
            artifacts["raw_activation_maps"] = str(npz_path.relative_to(output_dir))

            retrieval_payload, used_indices = serialize_retrieval(cache["retrieval"], phrases)
            retrieval_payload["text_attribute_gain_tanh"] = text_gain
            retrieval_path = class_dir / f"{image_id}_attribute_retrieval.json"
            write_json(retrieval_path, retrieval_payload)
            artifacts["attribute_retrieval"] = str(retrieval_path.relative_to(output_dir))

            feature_path = class_dir / f"{image_id}_recalled_attribute_features.npz"
            embedding_array = embeddings[used_indices].detach().float().cpu().numpy()
            np.savez_compressed(
                feature_path,
                cluster_indices=used_indices,
                embeddings=embedding_array,
            )
            artifacts["recalled_attribute_features"] = str(feature_path.relative_to(output_dir))

            target_retrieval = retrieval_payload["classes"][gt_id]
            wrong_retrieval = retrieval_payload["classes"][wrong_id]
            item["visualization"] = {
                "image_id": image_id,
                "pair_error_pixels": sample["pair_error_pixels"],
                "pair_error_fraction_of_gt": sample["pair_error_fraction_of_gt"],
                "gt_attribute_retrieval": target_retrieval,
                "wrong_attribute_retrieval": wrong_retrieval,
                "enhanced_classes": retrieval_payload["enhanced_classes"],
                "artifacts": artifacts,
            }
            print(
                f"Visualized {rank}/{len(summary['selected_classes'])}: "
                f"{gt_name} -> {wrong_name}, image {image_id}"
            )
    finally:
        capture.close()


def write_report(summary: Mapping[str, Any], output_dir: Path) -> None:
    lines = [
        "# Pascal VOC-20 高混淆类别与属性增强激活分析",
        "",
        f"- 验证图片：{summary['evaluated_images']} 张",
        "- 排名：按各 GT 类别的非对角混淆像素总数降序。",
        "- 样本：先确定该 GT 类别最常被误分成的类别，再选择该混淆对错误像素最多的图片。",
        "- 特征推理：与验证一致的 4 个 384×384 滑窗和 1 个全局视图，属性增强参数与验证配置一致。",
        "- cost-volume：使用属性增强后的文本特征。",
        "- 最终类别分数：捕获 Aggregator 输出的 raw logits，并严格按验证代码完成插值、sigmoid、局部窗口 fold 与全局视图平均；热力图为送入 `sem_seg_postprocess` 前的类别 mask score。",
        "- 原始 pre-sigmoid logits、384×384 插值 logits 及各视图 mask score 均保存在 NPZ。",
        "",
        "## 五个高混淆类别",
        "",
        "| 排名 | GT 类别 | 主要错误类别 | 总 FN 像素 | 该混淆对像素 | 最严重图片 | 图片内该混淆像素 |",
        "|---:|---|---|---:|---:|---|---:|",
    ]
    for item in summary["selected_classes"]:
        sample = item["selected_sample"]
        lines.append(
            f"| {item['rank']} | {item['gt_class']} | {item['wrong_class']} | "
            f"{item['false_negative_pixels']:,} | {item['dominant_pair_pixels']:,} | "
            f"{sample['image_id']} | {sample['pair_error_pixels']:,} |"
        )
    lines.extend(["", "## 可视化", ""])
    for item in summary["selected_classes"]:
        visual = item.get("visualization")
        if not visual:
            continue
        overview = visual["artifacts"]["overview"]
        lines.extend(
            [
                f"### {item['rank']}. {item['gt_class']} → {item['wrong_class']}",
                "",
                f"验证图片 `{visual['image_id']}`，该混淆对错误像素 "
                f"{visual['pair_error_pixels']:,}（占该图 GT 类别像素 "
                f"{visual['pair_error_fraction_of_gt'] * 100:.2f}%）。",
                "",
                f"![{item['gt_class']} to {item['wrong_class']}]({overview})",
                "",
            ]
        )
    lines.extend(
        [
            "## 文件说明",
            "",
            "- `confusion_matrix_row_normalized.png`：按 GT 行归一化的混淆矩阵。",
            "- `confusion_matrix_log_counts.png`：像素计数经 log 缩放的混淆矩阵。",
            "- `confusion_matrix_counts.csv`：原始像素计数。",
            "- `summary.json`：排名、样本选择、运行配置和可视化索引。",
            "- `high_confusion_classes/*/*_overview.png`：每个混淆对的八宫格总览。",
            "- `*_activation_maps.npz`：拼接后及五个视图的原始激活数组。",
            "- `*_attribute_retrieval.json`：所有被增强类别及五个视图的属性短语、索引、分数和置信度。",
            "- `*_recalled_attribute_features.npz`：召回聚类对应的真实属性特征向量。",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--openclip-pretrained", required=True)
    parser.add_argument("--attribute-database", required=True)
    parser.add_argument("--class-json", required=True)
    parser.add_argument("--top-classes", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze_predictions(args)
    save_confusion_artifacts(summary, output_dir)
    visualize_selected(args, summary)
    summary["completed_at"] = datetime.now().astimezone().isoformat()
    write_json(output_dir / "summary.json", summary)
    write_report(summary, output_dir)
    print(f"Completed VOC confusion visualization: {output_dir}")


if __name__ == "__main__":
    main()

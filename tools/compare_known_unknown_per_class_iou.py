#!/usr/bin/env python3
"""Compare per-class IoU and COCO-Stuff-known/unknown subset mIoU."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DATASETS = (
    ("coco_stuff", "coco.json", "coco_stuff"),
    ("ade20k_150", "ade150.json", "ade20k_150"),
    ("ade20k_847", "ade847.json", "ade20k_847"),
    ("pascal_voc20", "voc20.json", "pascal_voc20"),
    ("pascal_voc20_background", "voc20b.json", "pascal_voc20_background"),
    ("pascal_context_59", "pc59.json", "pascal_context_59"),
    ("pascal_context_459", "pc459.json", "pascal_context_459"),
)

# Auditable equivalence groups for cross-dataset spelling and naming variants.
# Comma-separated aliases already present in ADE class names are handled below.
SYNONYM_GROUPS: Sequence[Sequence[str]] = (
    ("airplane", "aeroplane"),
    ("motorcycle", "motorbike"),
    ("couch", "sofa", "settee"),
    ("tv", "television", "tv monitor", "tvmonitor"),
    ("dining table", "diningtable"),
    ("potted plant", "pottedplant"),
    ("sidewalk", "pavement"),
    ("shelf", "shelves"),
    ("railroad", "railway", "railway track", "track"),
    ("refrigerator", "fridge"),
    ("cell phone", "cellphone", "mobile phone", "mobilephone"),
    ("hair drier", "hair dryer", "hairdryer"),
    ("sports ball", "ball"),
    ("fire hydrant", "firehydrant"),
    ("traffic light", "trafficlight", "signal light", "signallight"),
    ("parking meter", "parkingmeter"),
    ("stop sign", "stopsign"),
    ("hot dog", "hotdog"),
    ("teddy bear", "teddybear"),
    ("wine glass", "wineglass"),
    ("baseball bat", "baseballbat"),
    ("baseball glove", "baseballglove"),
    ("window blind", "window blinds", "windowblind", "windowblinds"),
    ("playing field", "playingfield", "field"),
    ("ground", "earth"),
    ("window", "windowpane"),
    ("bag", "handbag"),
    ("computer", "laptop"),
    ("clothing", "clothes"),
    ("cupboard", "cabinet"),
    ("rug", "carpet"),
    ("rock", "stone"),
    ("stairs", "stair", "staircase"),
    ("rail", "railing"),
    ("sea", "ocean"),
    ("mountain", "mount"),
    ("fence", "fencing"),
    ("floor", "flooring"),
    ("building", "edifice"),
    ("road", "route"),
    ("plant", "flora", "plant life"),
    ("chair", "seat"),
    ("bedclothes", "bedding", "blanket"),
)

GENERIC_COCO_SUFFIXES = ("other", "stuff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--attribute-root", type=Path, required=True)
    parser.add_argument("--attribute-coco-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize_alias(value: str) -> str:
    value = value.strip().lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "", value)


def split_aliases(class_name: str) -> List[str]:
    return [item.strip() for item in class_name.split(",") if item.strip()]


def build_synonym_lookup() -> Dict[str, set]:
    lookup: Dict[str, set] = {}
    for group in SYNONYM_GROUPS:
        normalized = {normalize_alias(item) for item in group}
        for item in normalized:
            lookup.setdefault(item, set()).update(normalized)
    return lookup


SYNONYM_LOOKUP = build_synonym_lookup()


def expand_aliases(class_name: str, strip_generic_suffix: bool = False) -> set:
    aliases = {normalize_alias(item) for item in split_aliases(class_name)}
    aliases.discard("")
    if strip_generic_suffix:
        for alias in list(aliases):
            for suffix in GENERIC_COCO_SUFFIXES:
                if alias.endswith(suffix) and len(alias) > len(suffix):
                    aliases.add(alias[: -len(suffix)])
    expanded = set(aliases)
    for alias in aliases:
        expanded.update(SYNONYM_LOOKUP.get(alias, ()))
    return expanded


def build_known_vocabulary(coco_classes: Sequence[str]) -> Dict[str, List[str]]:
    vocabulary: Dict[str, List[str]] = {}
    for coco_class in coco_classes:
        for alias in expand_aliases(coco_class, strip_generic_suffix=True):
            vocabulary.setdefault(alias, []).append(coco_class)
    return vocabulary


def classify_known(
    class_name: str,
    known_vocabulary: Dict[str, List[str]],
) -> Tuple[bool, str, str]:
    for alias in sorted(expand_aliases(class_name)):
        matches = known_vocabulary.get(alias)
        if matches:
            return True, alias, matches[0]
    return False, "", ""


def load_last_sem_seg_metrics(log_path: Path) -> Dict[str, Optional[float]]:
    metric_lines = [
        line
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "OrderedDict([('sem_seg'" in line
    ]
    if not metric_lines:
        raise ValueError(f"No sem_seg OrderedDict found in {log_path}")
    line = metric_lines[-1]
    start = line.index("{")
    end = line.rfind("})])")
    if end < start:
        raise ValueError(f"Incomplete sem_seg metric line in {log_path}")
    payload = re.sub(r"\bnan\b", "None", line[start : end + 1])
    metrics = ast.literal_eval(payload)
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid sem_seg metric dictionary in {log_path}")
    return metrics


def effective_classes(dataset_name: str, class_json: Path) -> List[str]:
    classes = json.loads(class_json.read_text(encoding="utf-8"))
    if dataset_name == "pascal_voc20_background":
        # VOCbEvaluator maps every prediction >= 20 to one background channel.
        return classes[:20] + ["background"]
    return classes


def metric_value(metrics: Dict[str, Optional[float]], class_name: str) -> float:
    key = f"IoU-{class_name}"
    if key not in metrics:
        raise KeyError(f"Missing {key!r} in evaluator metrics")
    value = metrics[key]
    return float("nan") if value is None else float(value)


def valid_mean(values: Iterable[float]) -> Optional[float]:
    valid = [value for value in values if not math.isnan(value)]
    return mean(valid) if valid else None


def fmt(value: Optional[float], signed: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_dataset_dir = args.output_dir / "per_dataset"
    per_dataset_dir.mkdir(parents=True, exist_ok=True)

    coco_classes = json.loads(
        (args.datasets_dir / "coco.json").read_text(encoding="utf-8")
    )
    known_vocabulary = build_known_vocabulary(coco_classes)

    all_rows = []
    summary_rows = []
    classification = {}

    for dataset_name, class_json_name, log_dir_name in DATASETS:
        classes = effective_classes(
            dataset_name,
            args.datasets_dir / class_json_name,
        )
        baseline_log = args.baseline_root / log_dir_name / "log.txt"
        attribute_log = (
            args.attribute_coco_log
            if dataset_name == "coco_stuff"
            else args.attribute_root / log_dir_name / "log.txt"
        )
        baseline = load_last_sem_seg_metrics(baseline_log)
        attribute = load_last_sem_seg_metrics(attribute_log)

        dataset_rows = []
        for class_index, class_name in enumerate(classes):
            known, matched_alias, matched_coco_class = classify_known(
                class_name,
                known_vocabulary,
            )
            baseline_iou = metric_value(baseline, class_name)
            attribute_iou = metric_value(attribute, class_name)
            delta = (
                attribute_iou - baseline_iou
                if not math.isnan(baseline_iou) and not math.isnan(attribute_iou)
                else float("nan")
            )
            row = {
                "dataset": dataset_name,
                "class_index": class_index,
                "class_name": class_name,
                "split": "known" if known else "unknown",
                "matched_alias": matched_alias,
                "matched_coco_class": matched_coco_class,
                "baseline_iou": baseline_iou,
                "attribute_iou": attribute_iou,
                "delta_iou": delta,
            }
            dataset_rows.append(row)
            all_rows.append(row)

        classification[dataset_name] = {
            "known": [row["class_name"] for row in dataset_rows if row["split"] == "known"],
            "unknown": [row["class_name"] for row in dataset_rows if row["split"] == "unknown"],
        }

        for split_name in ("known", "unknown", "all"):
            selected = (
                dataset_rows
                if split_name == "all"
                else [row for row in dataset_rows if row["split"] == split_name]
            )
            baseline_mean = valid_mean(row["baseline_iou"] for row in selected)
            attribute_mean = valid_mean(row["attribute_iou"] for row in selected)
            delta_mean = (
                attribute_mean - baseline_mean
                if baseline_mean is not None and attribute_mean is not None
                else None
            )
            summary_rows.append({
                "dataset": dataset_name,
                "split": split_name,
                "class_count": len(selected),
                "baseline_valid_count": sum(
                    not math.isnan(row["baseline_iou"]) for row in selected
                ),
                "attribute_valid_count": sum(
                    not math.isnan(row["attribute_iou"]) for row in selected
                ),
                "baseline_miou": baseline_mean,
                "attribute_miou": attribute_mean,
                "delta_miou": delta_mean,
            })

        write_csv(
            per_dataset_dir / f"{dataset_name}.csv",
            dataset_rows,
            tuple(dataset_rows[0]),
        )

    for split_name in ("known", "unknown", "all"):
        selected = [
            row
            for row in summary_rows
            if row["split"] == split_name
            and row["baseline_miou"] is not None
            and row["attribute_miou"] is not None
        ]
        baseline_mean = mean(row["baseline_miou"] for row in selected)
        attribute_mean = mean(row["attribute_miou"] for row in selected)
        summary_rows.append({
            "dataset": "macro_average",
            "split": split_name,
            "class_count": sum(row["class_count"] for row in selected),
            "baseline_valid_count": sum(row["baseline_valid_count"] for row in selected),
            "attribute_valid_count": sum(row["attribute_valid_count"] for row in selected),
            "baseline_miou": baseline_mean,
            "attribute_miou": attribute_mean,
            "delta_miou": attribute_mean - baseline_mean,
        })

    for split_name in ("known", "unknown", "all"):
        selected = [
            row
            for row in all_rows
            if split_name == "all" or row["split"] == split_name
        ]
        baseline_mean = valid_mean(row["baseline_iou"] for row in selected)
        attribute_mean = valid_mean(row["attribute_iou"] for row in selected)
        summary_rows.append({
            "dataset": "pooled_class_average",
            "split": split_name,
            "class_count": len(selected),
            "baseline_valid_count": sum(
                not math.isnan(row["baseline_iou"]) for row in selected
            ),
            "attribute_valid_count": sum(
                not math.isnan(row["attribute_iou"]) for row in selected
            ),
            "baseline_miou": baseline_mean,
            "attribute_miou": attribute_mean,
            "delta_miou": (
                attribute_mean - baseline_mean
                if baseline_mean is not None and attribute_mean is not None
                else None
            ),
        })

    per_class_fields = tuple(all_rows[0])
    summary_fields = tuple(summary_rows[0])
    write_csv(args.output_dir / "per_class_iou.csv", all_rows, per_class_fields)
    write_csv(
        args.output_dir / "known_unknown_summary.csv",
        summary_rows,
        summary_fields,
    )

    for method_name, iou_key in (
        ("baseline", "baseline_iou"),
        ("attribute", "attribute_iou"),
    ):
        method_class_rows = [
            {
                "dataset": row["dataset"],
                "class_index": row["class_index"],
                "class_name": row["class_name"],
                "split": row["split"],
                "matched_alias": row["matched_alias"],
                "matched_coco_class": row["matched_coco_class"],
                "iou": row[iou_key],
            }
            for row in all_rows
        ]
        method_summary_rows = [
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "class_count": row["class_count"],
                "valid_count": row[f"{method_name}_valid_count"],
                "miou": row[f"{method_name}_miou"],
            }
            for row in summary_rows
        ]
        write_csv(
            args.output_dir / f"{method_name}_per_class_iou.csv",
            method_class_rows,
            tuple(method_class_rows[0]),
        )
        write_csv(
            args.output_dir / f"{method_name}_known_unknown_summary.csv",
            method_summary_rows,
            tuple(method_summary_rows[0]),
        )

    payload = {
        "definition": (
            "Known means at least one normalized class alias matches a COCO-Stuff "
            "class alias. ADE comma-separated aliases, generic COCO suffixes "
            "(-other/-stuff), and the audited synonym groups are included."
        ),
        "synonym_groups": SYNONYM_GROUPS,
        "classification": classification,
        "summary": summary_rows,
        "per_class": all_rows,
    }
    (args.output_dir / "known_unknown_results.json").write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 原版与属性融合版逐类别及已知/未知类别统计",
        "",
        "- 已知类：目标类别任一规范化别名与 COCO-Stuff 类别词表相交。",
        "- 别名来源：类别名中的逗号分隔别名、COCO 的 `-other/-stuff` 通用后缀以及脚本内显式同义词组。",
        "- 无有效验证像素的类别记为 NaN，不参与对应子集 mIoU。",
        "- VOC20-background 按实际评估通道统计：20 个前景类加 1 个 background。",
        "",
        "| 数据集 | 子集 | 类别数 | Base 有效类 | Attr 有效类 | Base mIoU | Attr mIoU | ΔmIoU |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        if row["dataset"] in ("macro_average", "pooled_class_average") or row["split"] != "all":
            lines.append(
                "| {dataset} | {split} | {class_count} | {baseline_valid_count} | "
                "{attribute_valid_count} | {base} | {attr} | {delta} |".format(
                    **row,
                    base=fmt(row["baseline_miou"]),
                    attr=fmt(row["attribute_miou"]),
                    delta=fmt(row["delta_miou"], signed=True),
                )
            )

    lines.extend(["", "## 类别归属", ""])
    for dataset_name, groups in classification.items():
        lines.extend([
            f"### {dataset_name}",
            "",
            f"- 已知类（{len(groups['known'])}）：" + ", ".join(groups["known"]),
            f"- 未知类（{len(groups['unknown'])}）：" + (
                ", ".join(groups["unknown"]) if groups["unknown"] else "无"
            ),
            "",
        ])
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare PCA-Seg baseline and ExCEL all-dataset summary files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


METRICS = ("mIoU", "fwIoU", "mACC", "pACC")


def parse_summary(path: Path):
    current = None
    expect_values = False
    results = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        section = re.match(r"^=====\s+([^\s]+)\s+\(", line)
        if section:
            current = section.group(1)
            expect_values = False
            continue
        if "copypaste: mIoU,fwIoU,mACC,pACC" in line:
            expect_values = True
            continue
        if current and expect_values and "copypaste:" in line:
            values = line.rsplit("copypaste:", 1)[1].strip().split(",")
            if len(values) != len(METRICS):
                raise ValueError(f"Invalid metric line in {path}: {line}")
            results[current] = {
                metric: float(value)
                for metric, value in zip(METRICS, values)
            }
            expect_values = False
    if not results:
        raise ValueError(f"No metrics found in {path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--attribute", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--attribute-database", required=True)
    parser.add_argument("--baseline-gpus", default="existing-result")
    parser.add_argument("--attribute-gpus", default="1,2,3,4")
    parser.add_argument(
        "--comparison-note",
        default=(
            "当前 ExCEL TSE 直接叠加到已训练好的原版权重，"
            "没有经过属性融合训练或校准。"
        ),
    )
    args = parser.parse_args()

    baseline = parse_summary(args.baseline)
    attribute = parse_summary(args.attribute)
    if baseline.keys() != attribute.keys():
        raise ValueError(
            f"Dataset mismatch: baseline={list(baseline)}, "
            f"attribute={list(attribute)}"
        )

    rows = []
    for dataset in baseline:
        row = {"dataset": dataset}
        for metric in METRICS:
            base = baseline[dataset][metric]
            attr = attribute[dataset][metric]
            row[f"baseline_{metric}"] = base
            row[f"attribute_{metric}"] = attr
            row[f"delta_{metric}"] = attr - base
            row[f"relative_{metric}_percent"] = (
                100.0 * (attr - base) / base if base else 0.0
            )
        rows.append(row)

    macro = {"dataset": "macro_average"}
    for metric in METRICS:
        for prefix in ("baseline", "attribute", "delta"):
            key = f"{prefix}_{metric}"
            macro[key] = sum(row[key] for row in rows) / len(rows)
        relative_key = f"relative_{metric}_percent"
        macro[relative_key] = (
            sum(row[relative_key] for row in rows) / len(rows)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["dataset"]
    for metric in METRICS:
        fields.extend([
            f"baseline_{metric}",
            f"attribute_{metric}",
            f"delta_{metric}",
            f"relative_{metric}_percent",
        ])
    with open(args.output_dir / "comparison.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows + [macro])

    payload = {
        "checkpoint": args.checkpoint,
        "attribute_database": args.attribute_database,
        "baseline_physical_gpus": args.baseline_gpus,
        "attribute_physical_gpus": args.attribute_gpus,
        "metrics": list(METRICS),
        "datasets": rows,
        "macro_average": macro,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# PCA-Seg 与 ExCEL 属性融合全数据集对比",
        "",
        f"- 权重：`{args.checkpoint}`",
        f"- 属性库：`{args.attribute_database}`",
        f"- Baseline 物理 GPU：`{args.baseline_gpus}`",
        f"- 属性组物理 GPU：`{args.attribute_gpus}`",
        "- 两组使用相同数据集、滑窗和 pooling 设置；属性组只开启 ExCEL TSE。",
        "",
        "| 数据集 | Base mIoU | Attr mIoU | ΔmIoU | Base fwIoU | Attr fwIoU | ΔfwIoU | Base mACC | Attr mACC | ΔmACC | Base pACC | Attr pACC | ΔpACC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows + [macro]:
        lines.append(
            "| {dataset} | {baseline_mIoU:.4f} | {attribute_mIoU:.4f} | {delta_mIoU:+.4f} | "
            "{baseline_fwIoU:.4f} | {attribute_fwIoU:.4f} | {delta_fwIoU:+.4f} | "
            "{baseline_mACC:.4f} | {attribute_mACC:.4f} | {delta_mACC:+.4f} | "
            "{baseline_pACC:.4f} | {attribute_pACC:.4f} | {delta_pACC:+.4f} |".format(**row)
        )
    lines.extend([
        "",
        "## 结论",
        "",
        f"七数据集宏平均 mIoU 变化为 `{macro['delta_mIoU']:+.4f}`。",
        args.comparison_note,
    ])
    (args.output_dir / "COMPARISON.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

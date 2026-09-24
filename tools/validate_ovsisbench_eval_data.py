#!/usr/bin/env python3
"""Validate file pairing and label ranges for all OVSISBench evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


DATASETS = (
    "DLRSD_all_sem_seg",
    "iSAID_all_sem_seg",
    "Potsdam_all_sem_seg",
    "Vaihingen_all_sem_seg",
    "UDD5_all_sem_seg",
    "LoveDA_all_sem_seg",
    "uavid_all_sem_seg",
    "VDD_all_sem_seg",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/mnt/data6/yc/datasets/OVSISBenchDataset",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ["OVSISBENCH_DATASETS"] = args.dataset_root
    os.environ["DETECTRON2_DATASETS"] = args.dataset_root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from detectron2.data import DatasetCatalog, MetadataCatalog
    import cat_seg.data.datasets  # noqa: F401

    report = {}
    for dataset_name in DATASETS:
        records = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        class_count = len(metadata.stuff_classes)
        observed = set()
        invalid = set()
        non_grayscale = []
        for record in records:
            mask_path = Path(record["sem_seg_file_name"])
            mask = np.asarray(Image.open(mask_path))
            if mask.ndim != 2:
                non_grayscale.append(str(mask_path))
                continue
            values = set(int(value) for value in np.unique(mask))
            observed.update(values)
            invalid.update(
                value
                for value in values
                if value != 255 and not 0 <= value < class_count
            )
        if non_grayscale or invalid:
            raise ValueError(
                f"{dataset_name}: non_grayscale={non_grayscale[:3]}, "
                f"invalid_labels={sorted(invalid)}"
            )
        report[dataset_name] = {
            "images": len(records),
            "classes": class_count,
            "labels": sorted(observed),
        }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

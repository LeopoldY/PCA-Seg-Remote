#!/usr/bin/env python3
"""Prepare OVSISBench's iSAID labels for joint DLRSD+iSAID training."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image


# iSAID local labels: ship=0, then the remaining 14 iSAID classes.
# PCA-Seg union labels: DLRSD=0..16, shared ship=13, iSAID-only=17..30.
ISAID_TO_UNION = np.array(
    [13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    return parser.parse_args()


def _remap_one(task: Tuple[Path, Path]) -> None:
    source, destination = task
    if destination.is_file():
        return
    labels = np.asarray(Image.open(source))
    if labels.ndim != 2:
        raise ValueError(f"Expected a single-channel mask: {source}")
    invalid = (labels != 255) & (labels >= len(ISAID_TO_UNION))
    if invalid.any():
        values = np.unique(labels[invalid]).tolist()
        raise ValueError(f"Unexpected iSAID labels {values} in {source}")
    remapped = np.full(labels.shape, 255, dtype=np.uint8)
    valid = labels != 255
    remapped[valid] = ISAID_TO_UNION[labels[valid]]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    Image.fromarray(remapped, mode="L").save(temporary)
    os.replace(temporary, destination)


def _tasks(source_dir: Path, destination_dir: Path) -> Iterable[Tuple[Path, Path]]:
    for source in sorted(source_dir.glob("*.png")):
        yield source, destination_dir / source.name


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    required = [
        root / "DLRSD_split/train/imgs",
        root / "DLRSD_split/train/D2masks",
        root / "DLRSD_split/val/imgs",
        root / "DLRSD_split/val/D2masks",
        root / "iSAID_split/train/images",
        root / "iSAID_split/train/D2masks",
        root / "iSAID_split/val/images",
        root / "iSAID_split/val/D2masks",
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing extracted OVSISBench paths: " + ", ".join(missing))

    summary = {}
    for split in ("train", "val"):
        source_dir = root / "iSAID_split" / split / "D2masks"
        destination_dir = root / "iSAID_union" / split / "D2masks"
        work = list(_tasks(source_dir, destination_dir))
        if not work:
            raise RuntimeError(f"No iSAID masks found in {source_dir}")
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            list(executor.map(_remap_one, work))
        produced = len(list(destination_dir.glob("*.png")))
        if produced != len(work):
            raise RuntimeError(
                f"Incomplete {split} conversion: expected {len(work)}, got {produced}"
            )
        summary[split] = produced

    manifest = {
        "dataset_root": str(root),
        "isaid_to_union": ISAID_TO_UNION.tolist(),
        "ignore_label": 255,
        "mask_counts": summary,
    }
    manifest_path = root / "iSAID_union/pcaseg_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

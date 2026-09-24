"""Register the DLRSD+iSAID training split from OVSISBench for PCA-Seg."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg


DLRSD_CLASSES = [
    "airplane",
    "bare soil",
    "buildings",
    "cars",
    "chaparral",
    "court",
    "dock",
    "field",
    "grass",
    "mobile home",
    "pavement",
    "sand",
    "sea",
    "ship",
    "tanks",
    "trees",
    "water",
]

ISAID_CLASSES = [
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "swimming pool",
    "roundabout",
    "soccer ball field",
    "plane",
    "harbor",
]

# Keep the descriptor JSON order. The shared "ship" class occupies index 13.
OVSISBENCH_TRAIN_CLASSES = DLRSD_CLASSES + ISAID_CLASSES[1:]

# Deterministic colors used only by Detectron2 visualizers.
OVSISBENCH_TRAIN_COLORS = [
    [37 * index % 256, 17 * index % 256, 29 * index % 256]
    for index in range(len(OVSISBENCH_TRAIN_CLASSES))
]


def _metadata(classes: List[str]) -> Dict[str, object]:
    return {
        "stuff_dataset_id_to_contiguous_id": {
            index: index for index in range(len(classes))
        },
        "stuff_classes": classes,
        "stuff_colors": OVSISBENCH_TRAIN_COLORS[: len(classes)],
    }


def _load_pairs(
    image_dir: str,
    gt_dir: str,
    image_ext: str,
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    records = load_sem_seg(
        gt_dir,
        image_dir,
        gt_ext="png",
        image_ext=image_ext,
    )
    if limit is not None:
        records = records[:limit]
    return records


def _register_split(
    *,
    name: str,
    image_dir: Path,
    gt_dir: Path,
    image_ext: str,
    classes: List[str],
    limit: Optional[int] = None,
) -> None:
    DatasetCatalog.register(
        name,
        lambda x=str(image_dir), y=str(gt_dir), ext=image_ext, n=limit: (
            _load_pairs(x, y, ext, n)
        ),
    )
    MetadataCatalog.get(name).set(
        image_root=str(image_dir),
        sem_seg_root=str(gt_dir),
        evaluator_type="sem_seg",
        ignore_label=255,
        **_metadata(classes),
    )


def register_ovsisbench(root: str) -> None:
    root_path = Path(root)
    specifications = [
        (
            "dlrsd",
            "jpg",
            root_path / "DLRSD_split",
            root_path / "DLRSD_split",
        ),
        (
            "isaid",
            "png",
            root_path / "iSAID_split",
            root_path / "iSAID_union",
        ),
    ]
    for dataset, image_ext, image_root, label_root in specifications:
        image_folder = "imgs" if dataset == "dlrsd" else "images"
        for split in ("train", "val"):
            image_dir = image_root / split / image_folder
            gt_dir = label_root / split / "D2masks"
            _register_split(
                name=f"ovsisbench_{dataset}_{split}_sem_seg",
                image_dir=image_dir,
                gt_dir=gt_dir,
                image_ext=image_ext,
                classes=OVSISBENCH_TRAIN_CLASSES,
            )
            # Small deterministic subsets for end-to-end smoke tests.
            _register_split(
                name=f"ovsisbench_{dataset}_{split}_smoke_sem_seg",
                image_dir=image_dir,
                gt_dir=gt_dir,
                image_ext=image_ext,
                classes=OVSISBENCH_TRAIN_CLASSES,
                limit=8 if split == "train" else 4,
            )

    # Native label spaces matching the separate-training protocol in the
    # original RSKT-Seg ViT-B configs. DLRSD masks are already 0..16, while
    # iSAID uses its original 0..14 masks (not the 31-class remapped copies).
    native_specifications = [
        (
            "DLRSD",
            "jpg",
            root_path / "DLRSD_split",
            "imgs",
            DLRSD_CLASSES,
        ),
        (
            "iSAID",
            "png",
            root_path / "iSAID_split",
            "images",
            ISAID_CLASSES,
        ),
    ]
    for dataset, image_ext, dataset_root, image_folder, classes in native_specifications:
        for split in ("train", "val"):
            _register_split(
                name=f"{dataset}_{split}_sem_seg",
                image_dir=dataset_root / split / image_folder,
                gt_dir=dataset_root / split / "D2masks",
                image_ext=image_ext,
                classes=classes,
            )
            _register_split(
                name=f"{dataset}_{split}_smoke_sem_seg",
                image_dir=dataset_root / split / image_folder,
                gt_dir=dataset_root / split / "D2masks",
                image_ext=image_ext,
                classes=classes,
                limit=8 if split == "train" else 4,
            )


_ROOT = os.getenv(
    "OVSISBENCH_DATASETS",
    os.getenv("DETECTRON2_DATASETS", "datasets"),
)
register_ovsisbench(_ROOT)

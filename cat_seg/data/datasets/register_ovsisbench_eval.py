"""Register all eight OVSISBench evaluation datasets used by RSKT-Seg."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

from .register_ovsisbench import DLRSD_CLASSES, ISAID_CLASSES


Category = Tuple[str, Sequence[int]]

DATASET_CATEGORIES: Dict[str, List[Category]] = {
    "DLRSD": [(name, (0, 0, 0)) for name in DLRSD_CLASSES],
    "iSAID": [(name, (0, 0, 0)) for name in ISAID_CLASSES],
    "Potsdam": [
        ("Impervious surfaces", (255, 255, 255)),
        ("Building", (0, 0, 255)),
        ("Low vegetation", (0, 255, 255)),
        ("Tree", (0, 255, 0)),
        ("Car", (255, 255, 0)),
        ("Clutter/background", (255, 0, 0)),
    ],
    "Vaihingen": [
        ("Impervious surfaces", (255, 255, 255)),
        ("Building", (0, 0, 255)),
        ("Low vegetation", (0, 255, 255)),
        ("Tree", (0, 255, 0)),
        ("Car", (255, 255, 0)),
        ("Clutter/background", (255, 0, 0)),
    ],
    "UDD5": [
        ("Vegetation", (107, 142, 35)),
        ("Building", (102, 102, 156)),
        ("Road", (128, 64, 128)),
        ("Vehicle", (0, 0, 142)),
        ("Other", (0, 0, 0)),
    ],
    "LoveDA": [
        ("no-data", (255, 255, 255)),
        ("Background", (255, 248, 220)),
        ("Building", (100, 149, 237)),
        ("Road", (102, 205, 170)),
        ("Water", (205, 133, 63)),
        ("Barren", (160, 32, 240)),
        ("Forest", (255, 64, 64)),
        ("Agriculture", (139, 69, 19)),
    ],
    "uavid": [
        ("Background clutter", (0, 0, 0)),
        ("Building", (128, 0, 0)),
        ("Road", (128, 64, 128)),
        ("Tree", (0, 128, 0)),
        ("Low vegetation", (128, 128, 0)),
        ("Moving car", (64, 0, 128)),
        ("Static car", (192, 0, 192)),
        ("Human", (64, 64, 0)),
    ],
    "VDD": [
        ("other", (0, 0, 0)),
        ("wall", (128, 0, 0)),
        ("road", (128, 64, 128)),
        ("vegetation", (0, 128, 0)),
        ("vehicle", (64, 0, 128)),
        ("roof", (192, 192, 128)),
        ("water", (0, 0, 128)),
    ],
}


def _metadata(categories: Sequence[Category]) -> Dict[str, object]:
    return {
        "stuff_dataset_id_to_contiguous_id": {
            index: index for index in range(len(categories))
        },
        "stuff_classes": [name for name, _ in categories],
        "stuff_colors": [list(color) for _, color in categories],
    }


def _load_multiple(
    pairs: Sequence[Tuple[Path, Path, str]],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for image_dir, gt_dir, image_ext in pairs:
        records.extend(
            load_sem_seg(
                str(gt_dir),
                str(image_dir),
                gt_ext="png",
                image_ext=image_ext,
            )
        )
    return records


def _register(
    name: str,
    root: Path,
    pairs: Sequence[Tuple[str, str, str]],
    categories: Sequence[Category],
) -> None:
    resolved = [
        (root / image_dir, root / gt_dir, image_ext)
        for image_dir, gt_dir, image_ext in pairs
    ]
    DatasetCatalog.register(
        name,
        lambda dataset_pairs=resolved: _load_multiple(dataset_pairs),
    )
    MetadataCatalog.get(name).set(
        image_root=str(root),
        sem_seg_root=str(root),
        evaluator_type="sem_seg",
        ignore_label=255,
        **_metadata(categories),
    )


def register_ovsisbench_eval(root: str) -> None:
    root_path = Path(root)
    specifications = [
        ("DLRSD_all_sem_seg", [("DLRSD/imgs", "DLRSD/D2masks", "jpg")], "DLRSD"),
        ("iSAID_all_sem_seg", [("iSAID/imgs", "iSAID/D2masks", "png")], "iSAID"),
        ("Potsdam_all_sem_seg", [("Potsdam/imgs", "Potsdam/D2masks", "png")], "Potsdam"),
        ("Vaihingen_all_sem_seg", [("Vaihingen/imgs", "Vaihingen/D2masks", "png")], "Vaihingen"),
        (
            "UDD5_all_sem_seg",
            [
                ("UDD5/train/src", "UDD5/train/gt", "JPG"),
                ("UDD5/val/src", "UDD5/val/gt", "JPG"),
            ],
            "UDD5",
        ),
        ("LoveDA_all_sem_seg", [("LoveDA/images_png", "LoveDA/masks_png", "png")], "LoveDA"),
        ("uavid_all_sem_seg", [("uavid/Images", "uavid/Labels", "png")], "uavid"),
        ("VDD_all_sem_seg", [("VDD/src", "VDD/gt", "JPG")], "VDD"),
    ]
    for name, pairs, category_key in specifications:
        _register(name, root_path, pairs, DATASET_CATEGORIES[category_key])


_ROOT = os.getenv(
    "OVSISBENCH_DATASETS",
    os.getenv("DETECTRON2_DATASETS", "datasets"),
)
register_ovsisbench_eval(_ROOT)

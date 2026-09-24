"""Native, disjoint DLRSD/iSAID splits; labels are already contiguous D2masks."""
import json
import os
from pathlib import Path
from detectron2.data import DatasetCatalog, MetadataCatalog

ROOT = Path(os.environ.get("OVSISBENCH_DATASETS", "/mnt/data6/yc/datasets/OVSISBenchDataset"))
CLASS_ROOT = Path(__file__).resolve().parents[3] / "datasets"


def load_pairs(image_dir, mask_dir, suffix, limit=None):
    images = {p.stem: p for p in image_dir.glob("*." + suffix)}
    masks = {p.stem: p for p in mask_dir.glob("*.png")}
    if not images or images.keys() != masks.keys():
        raise ValueError(f"Missing/unpaired data: {image_dir} ({len(images)} images, {len(masks)} masks)")
    keys = sorted(images)
    if limit is not None:
        keys = keys[:limit]
    return [{"file_name": str(images[k]), "sem_seg_file_name": str(masks[k])} for k in keys]


for dataset, folder, suffix in [("DLRSD", "imgs", "jpg"), ("iSAID", "images", "png")]:
    classes = json.loads((CLASS_ROOT / (dataset + ".json")).read_text())
    for split in ("train", "val"):
        base = ROOT / (dataset + "_split") / split
        for extra, limit in [("", None), ("_smoke", 8 if split == "train" else 4)]:
            name = f"{dataset}_{split}{extra}_sem_seg"
            DatasetCatalog.register(name, lambda b=base, f=folder, s=suffix, n=limit: load_pairs(b/f, b/"D2masks", s, n))
            MetadataCatalog.get(name).set(stuff_classes=classes,
                stuff_dataset_id_to_contiguous_id={i:i for i in range(len(classes))},
                evaluator_type="sem_seg", ignore_label=255,
                image_root=str(base/folder), sem_seg_root=str(base/"D2masks"))

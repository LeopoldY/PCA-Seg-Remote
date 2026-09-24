#!/usr/bin/env python3
"""Split the joint RSKT-Seg descriptor JSON by training dataset."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "attributes_text" / "rskt_seg_train_descriptors.json"
DATASETS = {
    "DLRSD": {
        "classes": PROJECT_ROOT / "datasets" / "DLRSD.json",
        "output": PROJECT_ROOT / "attributes_text" / "DLRSD_train_descriptors.json",
    },
    "iSAID": {
        "classes": PROJECT_ROOT / "datasets" / "iSAID.json",
        "output": PROJECT_ROOT / "attributes_text" / "iSAID_train_descriptors.json",
    },
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    source = load_json(SOURCE)
    if not isinstance(source, dict):
        raise TypeError(f"{SOURCE} must contain a JSON object")

    expected_union: set[str] = set()
    split_outputs: dict[str, dict[str, list[str]]] = {}

    for dataset_name, paths in DATASETS.items():
        classes = load_json(paths["classes"])
        if not isinstance(classes, list) or not all(
            isinstance(class_name, str) for class_name in classes
        ):
            raise TypeError(f"{paths['classes']} must contain a list of class names")
        if len(classes) != len(set(classes)):
            raise ValueError(f"{paths['classes']} contains duplicate classes")

        missing = [class_name for class_name in classes if class_name not in source]
        if missing:
            raise ValueError(f"{dataset_name} classes missing from source: {missing}")

        descriptors = {class_name: source[class_name] for class_name in classes}
        for class_name, descriptions in descriptors.items():
            if not isinstance(descriptions, list) or len(descriptions) != 20:
                raise ValueError(
                    f"{class_name!r} must contain exactly 20 descriptions"
                )
            if len(descriptions) != len(set(descriptions)):
                raise ValueError(f"{class_name!r} contains duplicate descriptions")
            if not all(isinstance(description, str) for description in descriptions):
                raise TypeError(f"{class_name!r} contains a non-string description")
            too_long = [
                description
                for description in descriptions
                if len(description.split()) > 77
            ]
            if too_long:
                raise ValueError(f"{class_name!r} contains descriptions over 77 words")

        split_outputs[dataset_name] = descriptors
        expected_union.update(classes)

    source_classes = set(source)
    if source_classes != expected_union:
        raise ValueError(
            "Source classes do not match the union of DLRSD and iSAID: "
            f"missing={sorted(expected_union - source_classes)}, "
            f"extra={sorted(source_classes - expected_union)}"
        )

    for dataset_name, descriptors in split_outputs.items():
        output = DATASETS[dataset_name]["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(descriptors, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        description_count = sum(len(items) for items in descriptors.values())
        print(
            f"wrote {dataset_name}: {len(descriptors)} classes, "
            f"{description_count} descriptions -> {output}"
        )


if __name__ == "__main__":
    main()

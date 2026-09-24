#!/usr/bin/env python3
"""Safely export a model-only EVA checkpoint from a DeCLIP training checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default="")
    return parser.parse_args()


def load_weights_only(path: Path) -> dict:
    # DeCLIP stores NumPy scalar metadata beside the tensor state dict. Keep
    # PyTorch's restricted unpickler and allow only the concrete NumPy scalar
    # and dtype classes required by that metadata.
    safe_globals = [
        np.core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("DeCLIP checkpoint must be a dictionary")
    return checkpoint


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    checkpoint = load_weights_only(input_path)

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("DeCLIP checkpoint has no non-empty state_dict")
    invalid = [
        key
        for key, value in state_dict.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    ]
    if invalid:
        raise TypeError(f"state_dict contains non-tensor entries: {invalid[:5]}")

    clean_state_dict = {
        key.removeprefix("module."): value.detach().cpu()
        for key, value in state_dict.items()
    }
    required_prefixes = ("visual.", "text.")
    for prefix in required_prefixes:
        if not any(key.startswith(prefix) for key in clean_state_dict):
            raise ValueError(f"state_dict has no {prefix!r} parameters")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": clean_state_dict}, output_path)

    metadata = {
        "source": str(input_path),
        "output": str(output_path),
        "epoch": int(checkpoint["epoch"]) if "epoch" in checkpoint else None,
        "name": str(checkpoint.get("name", "")),
        "tensor_count": len(clean_state_dict),
    }
    metadata_path = (
        Path(args.metadata_output).resolve()
        if args.metadata_output
        else output_path.with_suffix(output_path.suffix + ".json")
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

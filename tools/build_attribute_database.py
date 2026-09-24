#!/usr/bin/env python3
"""Re-encode ExCEL's official descriptors and build its database unchanged."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cat_seg.modeling.attribute_fusion import build_excel_cluster_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an ExCEL-format attribute bank from an official ExCEL "
            "descriptor JSON using PCA-Seg's text encoder."
        )
    )
    parser.add_argument("--descriptors-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--model-checkpoint",
        default="",
        help=(
            "Optional PCA-Seg/Detectron2 checkpoint. When provided, restore the "
            "fine-tuned CLIP text tower before encoding descriptors."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("eva", "open_clip", "openai"),
        default="eva",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-clusters", type=int, required=True)
    return parser.parse_args()


def load_text_encoder(
    backend: str,
    model_name: str,
    checkpoint: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Callable[[List[str]], torch.Tensor]]:
    if backend == "eva":
        import cat_seg.src.open_clip as open_clip
        import cat_seg.src.open_clip.eva_clip as eva_clip

        model = eva_clip.create_model(
            model_name=model_name,
            pretrained=checkpoint or None,
            force_custom_clip=True,
            precision="fp32",
            device=str(device),
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    elif backend == "open_clip":
        import cat_seg.src.open_clip as open_clip

        # Text-only encoding does not need image transforms or training args.
        model = open_clip.create_model(
            model_name,
            pretrained=checkpoint or None,
            device=str(device),
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    else:
        from cat_seg.third_party import clip

        model, _ = clip.load(
            model_name,
            device=str(device),
            jit=False,
            checkpoint=checkpoint or None,
        )
        tokenizer = clip.tokenize
    return model.float().eval(), tokenizer


def restore_pca_seg_text_encoder(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> None:
    """Restore only the CLIP text tower from a PCA-Seg checkpoint."""
    if not checkpoint_path:
        return

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("model", checkpoint)
    prefix = "sem_seg_head.predictor.clip_model."
    text_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(f"{prefix}text.")
    }
    if not text_state:
        raise ValueError(
            "No PCA-Seg CLIP text-tower parameters were found under "
            f"{prefix!r} in {checkpoint_path}."
        )

    expected_text_keys = {
        key for key in model.state_dict() if key.startswith("text.")
    }
    missing_text_keys = sorted(expected_text_keys - set(text_state))
    unexpected_text_keys = sorted(set(text_state) - expected_text_keys)
    if missing_text_keys or unexpected_text_keys:
        raise ValueError(
            "PCA-Seg text tower is incompatible with the selected encoder: "
            f"missing={missing_text_keys[:5]}, "
            f"unexpected={unexpected_text_keys[:5]}."
        )

    incompatible = model.load_state_dict(text_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"Unexpected text-tower keys: {unexpected[:5]}")
    print(
        f"Restored {len(text_state)} text-tower tensors from "
        f"{checkpoint_path}"
    )


@torch.no_grad()
def encode_class_descriptors(
    descriptions: dict,
    model: torch.nn.Module,
    tokenizer: Callable[[List[str]], torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> List[torch.Tensor]:
    class_embeddings = []
    for class_name, class_descriptions in descriptions.items():
        if not isinstance(class_descriptions, list) or not class_descriptions:
            raise ValueError(f"Descriptor list is empty for class {class_name!r}.")
        sentences = [str(item).lower() for item in class_descriptions]
        batches = []
        for start in range(0, len(sentences), max(1, batch_size)):
            tokens = tokenizer(sentences[start:start + batch_size]).to(device)
            encoded = model.encode_text(tokens)
            batches.append(F.normalize(encoded.float(), p=2, dim=1).cpu())
        class_embeddings.append(torch.cat(batches))
    return class_embeddings


def main() -> None:
    args = parse_args()
    with open(args.descriptors_json, "r", encoding="utf-8") as stream:
        descriptions = json.load(stream)
    if not isinstance(descriptions, dict) or not descriptions:
        raise ValueError("ExCEL descriptors JSON must be a non-empty object.")

    device = torch.device(args.device)
    model, tokenizer = load_text_encoder(
        backend=args.backend,
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        device=device,
    )
    restore_pca_seg_text_encoder(model, args.model_checkpoint)
    class_embeddings = encode_class_descriptors(
        descriptions=descriptions,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
    )
    cluster_bank, class_flags = build_excel_cluster_bank(
        class_embeddings=class_embeddings,
        num_clusters=args.num_clusters,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save([cluster_bank, class_flags], output)
    print(
        f"Saved ExCEL database to {output}: "
        f"cluster_bank={tuple(cluster_bank.shape)}, "
        f"class_flags={tuple(class_flags.shape)}"
    )


if __name__ == "__main__":
    main()

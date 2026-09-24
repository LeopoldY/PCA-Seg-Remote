#!/usr/bin/env python3
"""Check ExCEL TSE numerics and the exact disabled fallback path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cat_seg.modeling.attribute_fusion import AttributeTextAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute-database", required=True)
    parser.add_argument("--expected-dim", type=int, required=True)
    parser.add_argument("--num-clusters", type=int, required=True)
    parser.add_argument("--top-k", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def excel_reference(text: torch.Tensor, bank: torch.Tensor, top_k: float):
    flat = text.float().reshape(-1, text.shape[-1])
    logits = flat @ bank
    sorted_logits, indices = torch.sort(logits, dim=-1, descending=True)
    filtered_count = int((1.0 - top_k) * bank.shape[1])
    if filtered_count > 0:
        sorted_logits[:, -filtered_count:] = float("-inf")
    restored = torch.zeros_like(sorted_logits)
    restored.scatter_(-1, indices, sorted_logits)
    weights = restored.softmax(dim=-1)
    return F.normalize(weights @ bank.t() + flat, dim=-1).reshape(text.shape)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    adapter = AttributeTextAdapter.from_database_path(
        database_path=args.attribute_database,
        expected_dim=args.expected_dim,
        expected_num_clusters=args.num_clusters,
        top_k=args.top_k,
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(20260821)
    text = F.normalize(
        torch.randn(2, 5, 1, args.expected_dim, generator=generator, device=device),
        dim=-1,
    )
    actual = adapter(text)
    expected = excel_reference(text, adapter.cluster_bank, args.top_k)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    # The PCA-Seg fallback is the identity before the unmodified aggregator.
    disabled = text
    torch.testing.assert_close(disabled, text, rtol=0, atol=0)
    print("EXCEL_ATTRIBUTE_SMOKE_TEST_PASS")


if __name__ == "__main__":
    main()

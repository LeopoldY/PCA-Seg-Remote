"""ExCEL Text Semantic Enrichment adapted only for PCA-Seg tensor shapes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .attribute_database import load_excel_attribute_database


class AttributeTextAdapter(nn.Module):
    """Apply ExCEL's attribute aggregation to PCA-Seg class prompts.

    ExCEL receives ``[T, D]`` text features. PCA-Seg receives ``[B, T, P, D]``;
    flattening the first three axes applies the same operation independently to
    every batch item, class and prompt without introducing another fusion rule.
    """

    def __init__(
        self,
        cluster_bank: Tensor,
        class_flags: Tensor,
        top_k: Optional[float] = 0.9,
    ) -> None:
        super().__init__()
        if cluster_bank.ndim != 2:
            raise ValueError("cluster_bank must have ExCEL shape [D, M].")
        if top_k is not None and not 0.0 <= top_k <= 1.0:
            raise ValueError("top_k must be in [0, 1] or None.")
        self.top_k = top_k
        self.register_buffer(
            "cluster_bank", cluster_bank.detach().float(), persistent=False
        )
        self.register_buffer(
            "class_flags", class_flags.detach().float(), persistent=False
        )

    @classmethod
    def from_database_path(
        cls,
        database_path: str,
        expected_dim: int,
        expected_num_clusters: Optional[int] = None,
        top_k: Optional[float] = 0.9,
    ) -> "AttributeTextAdapter":
        cluster_bank, class_flags = load_excel_attribute_database(
            database_path=database_path,
            expected_dim=expected_dim,
            expected_num_clusters=expected_num_clusters,
        )
        return cls(cluster_bank, class_flags, top_k=top_k)

    def forward(self, text_features: Tensor) -> Tensor:
        if text_features.ndim != 4:
            raise ValueError("text_features must have shape [B, T, P, D].")
        if text_features.shape[-1] != self.cluster_bank.shape[0]:
            raise ValueError("Text and ExCEL attribute dimensions do not match.")

        original_shape = text_features.shape
        original_dtype = text_features.dtype
        foreground_text = text_features.float().reshape(
            -1, original_shape[-1]
        )
        attribute_bank = self.cluster_bank.to(foreground_text.device)

        # This is ExCEL's attr_aggregate operation, including its topK rule.
        correlation = (foreground_text @ attribute_bank).softmax(dim=-1)
        if self.top_k is not None:
            filtered_count = int(
                (1.0 - self.top_k) * attribute_bank.shape[1]
            )
            attention_logits = foreground_text @ attribute_bank
            sorted_logits, indices = torch.sort(
                attention_logits, dim=-1, descending=True
            )
            if filtered_count > 0:
                sorted_logits[:, -filtered_count:] = float("-inf")
            restored_logits = torch.zeros_like(sorted_logits)
            restored_logits.scatter_(-1, indices, sorted_logits)
            correlation = restored_logits.softmax(dim=-1)

        enriched_text = (
            correlation @ attribute_bank.t() + foreground_text
        )
        enriched_text = F.normalize(enriched_text, dim=-1)
        return enriched_text.reshape(original_shape).to(original_dtype)

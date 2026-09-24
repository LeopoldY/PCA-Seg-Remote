"""Lightweight sparse Transformer fusion for PCA-Seg feature branches.

The module treats the spatial-aggregation and class-aggregation results as a
two-token source sequence.  A single-query cross-attention layer selects and
combines the two sources independently for every class/spatial location.  A
Switch/DeepSeek-style shared-plus-routed FFN then conditionally transforms the
fused token.

No attention is performed across classes or spatial locations here; those
relations remain the responsibility of PCA-Seg's Class Transformer and Swin
blocks respectively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """A compact Transformer FFN expert."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DualRouteSparseTransformer(nn.Module):
    """Fuse spatial and class features with two-source attention and sparse MoE.

    Args:
        dim: Input and output channel dimension.
        num_heads: Number of heads in the two-source cross-attention.
        num_routed_experts: Number of conditionally activated FFN experts.
        top_k: Number of routed experts per token.  Only Top-1 is intentionally
            supported to keep dispatch simple and computation lightweight.
        expert_ratio: Hidden width of each SwiGLU expert relative to ``dim``.
        attention_dropout: Dropout used by attention and output projections.
        router_balance_weight: Weight applied to the standard expert load
            balancing objective returned during training.

    Input/output shape:
        ``[B, C, T, H, W]`` -> ``[B, C, T, H, W]``.
    """

    def __init__(
        self,
        dim,
        num_heads=4,
        num_routed_experts=4,
        top_k=1,
        expert_ratio=1.0,
        attention_dropout=0.0,
        router_balance_weight=0.01,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_routed_experts <= 0:
            raise ValueError("num_routed_experts must be positive")
        if top_k != 1:
            raise ValueError("DualRouteSparseTransformer currently supports TOP_K=1 only")
        if expert_ratio <= 0:
            raise ValueError("expert_ratio must be positive")
        if router_balance_weight < 0:
            raise ValueError("router_balance_weight must be non-negative")

        self.dim = dim
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.router_balance_weight = router_balance_weight

        # Separate projections preserve the identity of the two feature routes
        # without adding hand-crafted route embeddings.
        self.spatial_proj = nn.Linear(dim, dim)
        self.class_proj = nn.Linear(dim, dim)
        self.source_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.expert_norm = nn.LayerNorm(dim)
        self.router = nn.Linear(dim, num_routed_experts, bias=False)
        expert_hidden_dim = max(1, int(round(dim * expert_ratio)))
        self.shared_expert = SwiGLUExpert(dim, expert_hidden_dim)
        self.routed_experts = nn.ModuleList(
            [
                SwiGLUExpert(dim, expert_hidden_dim)
                for _ in range(num_routed_experts)
            ]
        )
        self.output_proj = nn.Linear(dim, dim)
        self.output_dropout = nn.Dropout(attention_dropout)

    def _route(self, x):
        """Run only the selected routed expert for each flattened token."""
        # Keeping the small router in FP32 follows common sparse-MoE practice
        # and avoids unstable expert choices under reduced-precision training.
        router_logits = F.linear(
            x.float(),
            self.router.weight.float(),
            None,
        )
        router_probs_fp32 = F.softmax(router_logits, dim=-1)
        route_scores_fp32, route_indices = router_probs_fp32.max(dim=-1)
        route_scores = route_scores_fp32.to(dtype=x.dtype)

        routed_output = torch.zeros_like(x)
        for expert_index, expert in enumerate(self.routed_experts):
            token_mask = route_indices == expert_index
            if token_mask.any():
                expert_output = expert(x[token_mask])
                routed_output[token_mask] = (
                    expert_output * route_scores[token_mask].unsqueeze(-1)
                )

        # Keep every expert in the autograd graph even if a small local batch
        # happens not to route a token to it.  This avoids DDP unused-parameter
        # failures without evaluating inactive experts.
        zero_dependency = x.new_zeros(())
        for expert in self.routed_experts:
            for parameter in expert.parameters():
                zero_dependency = zero_dependency + parameter.reshape(-1)[0] * 0.0
        routed_output = routed_output + zero_dependency

        route_fraction = F.one_hot(
            route_indices,
            num_classes=self.num_routed_experts,
        ).float().mean(dim=0)
        mean_route_probability = router_probs_fp32.mean(dim=0)
        balance_loss = self.num_routed_experts * torch.sum(
            route_fraction * mean_route_probability
        )
        return routed_output, balance_loss

    def forward(
        self,
        spatial_feat,
        class_feat,
        batched_inputs=None,
        return_aux_loss=False,
    ):
        del batched_inputs  # Kept for drop-in compatibility with DualFeatureMoE.

        if spatial_feat.shape != class_feat.shape:
            raise ValueError(
                "spatial_feat and class_feat must have identical shapes, got "
                f"{tuple(spatial_feat.shape)} and {tuple(class_feat.shape)}"
            )
        if spatial_feat.ndim != 5:
            raise ValueError(
                "Expected [B, C, T, H, W] inputs, got "
                f"{tuple(spatial_feat.shape)}"
            )

        batch, channels, classes, height, width = spatial_feat.shape
        if channels != self.dim:
            raise ValueError(
                f"Expected {self.dim} input channels, got {channels}"
            )

        spatial_tokens = spatial_feat.permute(0, 2, 3, 4, 1).reshape(
            -1, channels
        )
        class_tokens = class_feat.permute(0, 2, 3, 4, 1).reshape(
            -1, channels
        )
        spatial_tokens = self.spatial_proj(spatial_tokens)
        class_tokens = self.class_proj(class_tokens)

        source_tokens = torch.stack(
            [spatial_tokens, class_tokens],
            dim=1,
        )
        source_tokens = self.source_norm(source_tokens)
        query = 0.5 * (spatial_tokens + class_tokens)
        attention_output, _ = self.cross_attention(
            self.query_norm(query).unsqueeze(1),
            source_tokens,
            source_tokens,
            need_weights=False,
        )
        fused_tokens = query + self.output_dropout(
            attention_output.squeeze(1)
        )

        expert_input = self.expert_norm(fused_tokens)
        shared_output = self.shared_expert(expert_input)
        routed_output, balance_loss = self._route(expert_input)
        fused_tokens = fused_tokens + self.output_dropout(
            self.output_proj(shared_output + routed_output)
        )

        output = fused_tokens.reshape(
            batch,
            classes,
            height,
            width,
            channels,
        ).permute(0, 4, 1, 2, 3).contiguous()

        if return_aux_loss:
            return output, self.router_balance_weight * balance_loss
        return output

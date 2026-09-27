"""Learned local fusion of spatial and class-conditioned features.

The fixed 4C feed-forward width matches the original four-expert
DualFeatureMoE budget. See DUAL_FEATURE_MOE_LEARNED_FUSION_PROPOSAL.md.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LearnedDualFeatureFusion(nn.Module):
    """Fuse two [B, C, T, H, W] tensors with parameters shared across T.

    A local convolutional gate selects branch contributions per channel and
    position. A full two-branch linear bypass and a residual convolutional
    gated feed-forward network retain and refine the combined information.
    """

    def __init__(self, dim):
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        hidden_dim = 4 * dim

        self.spatial_proj = nn.Conv2d(dim, dim, 1)
        self.class_proj = nn.Conv2d(dim, dim, 1)
        self.gate_dw = nn.Conv2d(
            2 * dim, 2 * dim, 3, padding=1, groups=2 * dim
        )
        self.gate_in = nn.Conv2d(2 * dim, dim, 1)
        self.gate_out = nn.Conv2d(dim, dim, 1)
        self.value_proj = nn.Conv2d(dim, dim, 1)

        self.residual = nn.Conv2d(2 * dim, dim, 1)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

        # Normalize channels independently at every class/spatial position.
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.expand = nn.Conv2d(dim, 2 * hidden_dim, 1)
        self.ffn_dw = nn.Conv2d(
            2 * hidden_dim, 2 * hidden_dim, 3,
            padding=1, groups=2 * hidden_dim,
        )
        self.reduce = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, spatial_feat, class_feat, batched_inputs=None):
        del batched_inputs  # Retain the existing dense-fusion call signature.
        if spatial_feat.ndim != 5 or spatial_feat.shape != class_feat.shape:
            raise ValueError("Expected two identically shaped [B, C, T, H, W] tensors")
        if spatial_feat.device != class_feat.device:
            raise ValueError("Spatial and class features must be on the same device")
        if not spatial_feat.is_floating_point() or not class_feat.is_floating_point():
            raise ValueError("Fusion inputs must be floating point tensors")
        batch, channels, classes, height, width = spatial_feat.shape
        if channels != self.dim or min(batch, classes, height, width) <= 0:
            raise ValueError(f"Expected {self.dim} channels and nonempty feature dimensions")

        spatial = spatial_feat.permute(0, 2, 1, 3, 4).reshape(
            batch * classes, channels, height, width
        )
        semantic = class_feat.permute(0, 2, 1, 3, 4).reshape(
            batch * classes, channels, height, width
        )
        spatial = self.spatial_proj(spatial)
        semantic = self.class_proj(semantic)
        joint = torch.cat((spatial, semantic), dim=1)

        gate = torch.sigmoid(self.gate_out(
            F.gelu(self.gate_in(self.gate_dw(joint)))
        ))
        fused = self.value_proj(gate * spatial + (1 - gate) * semantic)
        fused = fused + self.res_scale * self.residual(joint)

        normalized = self.norm(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        first, second = self.ffn_dw(self.expand(normalized)).chunk(2, dim=1)
        output = fused + self.reduce(F.gelu(first) * second)
        return output.reshape(batch, classes, channels, height, width).permute(
            0, 2, 1, 3, 4
        )

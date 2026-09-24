"""AFF residual fusion with four densely evaluated, packed EPL experts.

AFF: Dai et al., Attentional Feature Fusion, WACV 2021, Eq. (4).
https://arxiv.org/abs/2009.14082

Only the original DualFeatureMoE linear residual is replaced by AFF. Experts
still consume both projected branches, and their convolutions are packed along
channels without sharing weights. AFF uses the paper's weighted average (no
extra factor of two). Pooling is over H/W, independently for every class.
"""

import torch
from torch import nn


class AFFResidualMoE(nn.Module):
    """Fuse two [B, C, T, H, W] branches, preserving the same output shape.

    Four experts are always evaluated. Their packed SyncBatchNorm preserves
    the legacy cross-worker statistics. AFF uses ordinary BatchNorm, as in
    its reference implementation, without additional collectives. Training
    its pooled path requires B*T > 1 on each worker.
    """

    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        if dim <= 0 or reduction_ratio <= 0 or dim < reduction_ratio:
            raise ValueError("Require dim >= reduction_ratio > 0")
        self.dim = dim
        self.experts = 4
        reduced_dim = dim // reduction_ratio

        self.spatial_proj = nn.Conv2d(dim, dim, 1)
        self.class_proj = nn.Conv2d(dim, dim, 1)

        # All experts read the same 2C input; only the second convolution
        # is grouped. Each group retains the legacy two-input/one-output map.
        self.packed_experts = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim * self.experts, 1),
            nn.SyncBatchNorm(2 * dim * self.experts),
            nn.GELU(),
            nn.Conv2d(
                2 * dim * self.experts,
                dim * self.experts,
                3,
                padding=1,
                groups=dim * self.experts,
            ),
            nn.GELU(),
        )
        self.gate_net = nn.Sequential(
            nn.Conv2d(2 * dim, reduced_dim, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_dim, self.experts, 1),
            nn.Softmax(dim=1),
        )

        def context_bottleneck():
            return nn.Sequential(
                nn.Conv2d(dim, reduced_dim, 1),
                nn.BatchNorm2d(reduced_dim),
                nn.ReLU(),
                nn.Conv2d(reduced_dim, dim, 1),
                nn.BatchNorm2d(dim),
            )

        self.local_att = context_bottleneck()
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), context_bottleneck()
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, spatial_feat, class_feat, batched_inputs=None):
        del batched_inputs  # Same interface as DualFeatureMoE.
        if spatial_feat.ndim != 5 or spatial_feat.shape != class_feat.shape:
            raise ValueError("Expected two identically shaped [B, C, T, H, W] tensors")
        batch, channels, classes, height, width = spatial_feat.shape
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {channels}")

        # Transformer outputs can retain channel-last strides. Normalize the
        # convolution layout once, avoiding mismatched DDP gradient buckets.
        spatial = spatial_feat.permute(0, 2, 1, 3, 4).reshape(
            batch * classes, channels, height, width
        ).contiguous()
        semantic = class_feat.permute(0, 2, 1, 3, 4).reshape_as(spatial).contiguous()
        spatial = self.spatial_proj(spatial)
        semantic = self.class_proj(semantic)
        expert_input = torch.cat((spatial, semantic), dim=1)

        expert_features = self.packed_experts(expert_input).reshape(
            batch * classes, self.experts, channels, height, width
        )
        expert_weights = self.gate_net(expert_input).unsqueeze(2)
        mixed = (expert_features * expert_weights).sum(dim=1)

        context = spatial + semantic
        branch_weight = torch.sigmoid(
            self.local_att(context) + self.global_att(context)
        )
        residual = branch_weight * spatial + (1.0 - branch_weight) * semantic
        fused = mixed + self.res_scale * residual
        return fused.reshape(batch, classes, channels, height, width).permute(
            0, 2, 1, 3, 4
        ).contiguous()

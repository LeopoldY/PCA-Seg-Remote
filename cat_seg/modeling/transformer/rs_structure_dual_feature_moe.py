"""Class-conditioned local structure fusion for remote sensing.

Four dense experts retain the DualFeatureMoE channel transforms and use
center, detail, region, and paired-direction operators in place of spatial
convolution kernels. See DUAL_FEATURE_MOE_RS_STRUCTURE_FUSION_STRATEGY.md.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RSStructureDualFeatureMoE(nn.Module):
    """Fuse two [B, C, T, H, W] tensors with a variable candidate vocabulary.

    Directions share the same rule. All neighborhood operations use replicate
    padding; no wraparound, class-index parameters, or teacher inputs are used.
    """

    # Consecutive offsets form opposite pairs: horizontal, vertical, diagonals.
    _OFFSETS = ((0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (-1, -1), (1, -1), (-1, 1))

    def __init__(self, dim, reduction_ratio=4, tau=0.1,
                 correction_max=0.5, correction_init=0.1,
                 detach_guidance=True):
        super().__init__()
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
               for v in (dim, reduction_ratio)) or dim < reduction_ratio:
            raise ValueError("Require integer dim >= reduction_ratio > 0")
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("TAU must be finite and positive")
        if (not math.isfinite(correction_max)
                or not math.isfinite(correction_init)
                or not 0 < correction_init < correction_max <= 0.5):
            raise ValueError("Require 0 < CORRECTION_INIT < CORRECTION_MAX <= 0.5")
        if not isinstance(detach_guidance, bool):
            raise ValueError("DETACH_GUIDANCE must be a bool")

        self.dim = dim
        self.experts = 4
        self.tau = float(tau)
        self.correction_max = float(correction_max)
        self.detach_guidance = detach_guidance
        self.eps = 1e-6
        reduced_dim = dim // reduction_ratio

        self.spatial_proj = nn.Conv2d(dim, dim, 1)
        self.class_proj = nn.Conv2d(dim, dim, 1)
        self.point_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, 1),
                nn.SyncBatchNorm(2 * dim),
                nn.GELU(),
                nn.Conv2d(2 * dim, dim, 1, groups=dim),
                nn.GELU(),
            ) for _ in range(self.experts)
        ])
        self.gate_net = nn.Sequential(
            nn.Conv2d(2 * dim + 4, reduced_dim, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_dim, self.experts, 1),
            nn.Softmax(dim=1),
        )
        # Start with feature-only routing; descriptors enter through learning.
        with torch.no_grad():
            self.gate_net[0].weight[:, 2 * dim:].zero_()

        initial_logit = math.log(correction_init / (correction_max - correction_init))
        self.correction_logits = nn.Parameter(torch.full((3,), initial_logit))
        self.residual = nn.Conv2d(2 * dim, dim, 1)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _neighbor(padded, height, width, offset):
        dy, dx = offset
        return padded[..., 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]

    def _neighborhood_statistics(self, spatial, semantic):
        if self.detach_guidance:
            spatial, semantic = spatial.detach(), semantic.detach()
        # Disabling autocast explicitly keeps similarities and exponentials FP32.
        with torch.autocast(device_type=spatial.device.type, enabled=False):
            spatial = F.normalize(spatial.float(), dim=1, eps=self.eps)
            semantic = F.normalize(semantic.float(), dim=1, eps=self.eps)
            height, width = spatial.shape[-2:]
            spatial_pad = F.pad(spatial, (1, 1, 1, 1), mode="replicate")
            semantic_pad = F.pad(semantic, (1, 1, 1, 1), mode="replicate")
            affinities = []
            spatial_distances = []
            semantic_distances = []
            for offset in self._OFFSETS:
                s_neighbor = self._neighbor(spatial_pad, height, width, offset)
                q_neighbor = self._neighbor(semantic_pad, height, width, offset)
                s_cos = (spatial * s_neighbor).sum(dim=1, keepdim=True).clamp(-1, 1)
                q_cos = (semantic * q_neighbor).sum(dim=1, keepdim=True).clamp(-1, 1)
                s_dist = (1 - s_cos) * 0.5
                q_dist = (1 - q_cos) * 0.5
                spatial_distances.append(s_dist)
                semantic_distances.append(q_dist)
                affinities.append(torch.exp(-(s_dist + q_dist) / (2 * self.tau)))

            affinities = torch.cat(affinities, dim=1)
            paired = affinities[:, 0::2] * affinities[:, 1::2]
            descriptors = torch.cat((
                torch.cat(spatial_distances, dim=1).mean(dim=1, keepdim=True),
                torch.cat(semantic_distances, dim=1).mean(dim=1, keepdim=True),
                affinities.mean(dim=1, keepdim=True),
                paired.amax(dim=1, keepdim=True) - paired.amin(dim=1, keepdim=True),
            ), dim=1)
        return affinities, paired, descriptors

    @staticmethod
    def _mean3x3(feature):
        with torch.autocast(device_type=feature.device.type, enabled=False):
            mean = F.avg_pool2d(
                F.pad(feature.float(), (1, 1, 1, 1), mode="replicate"),
                kernel_size=3, stride=1,
            )
        return mean.to(dtype=feature.dtype)

    def _region_average(self, feature, affinities):
        with torch.autocast(device_type=feature.device.type, enabled=False):
            values = feature.float()
            height, width = values.shape[-2:]
            padded = F.pad(values, (1, 1, 1, 1), mode="replicate")
            numerator = values
            # Accumulate one offset at a time, without a [BT, C, 8, H, W] unfold.
            for i, offset in enumerate(self._OFFSETS):
                neighbor = self._neighbor(padded, height, width, offset)
                numerator = numerator + affinities[:, i:i + 1] * neighbor
            average = numerator / (1 + affinities.sum(dim=1, keepdim=True))
        return average.to(dtype=feature.dtype)

    def _paired_direction_average(self, feature, paired):
        with torch.autocast(device_type=feature.device.type, enabled=False):
            values = feature.float()
            height, width = values.shape[-2:]
            padded = F.pad(values, (1, 1, 1, 1), mode="replicate")
            numerator = values
            for i in range(4):
                positive = self._neighbor(padded, height, width, self._OFFSETS[2 * i])
                negative = self._neighbor(padded, height, width, self._OFFSETS[2 * i + 1])
                numerator = numerator + paired[:, i:i + 1] * ((positive + negative) * 0.5)
            average = numerator / (1 + paired.sum(dim=1, keepdim=True))
        return average.to(dtype=feature.dtype)

    def forward(self, spatial_feat, class_feat, batched_inputs=None):
        del batched_inputs
        if spatial_feat.ndim != 5 or spatial_feat.shape != class_feat.shape:
            raise ValueError("Expected two identically shaped [B, C, T, H, W] tensors")
        if spatial_feat.device != class_feat.device or spatial_feat.dtype != class_feat.dtype:
            raise ValueError("Spatial and class features must have the same device and dtype")
        batch, channels, classes, height, width = spatial_feat.shape
        if channels != self.dim or min(batch, classes, height, width) <= 0:
            raise ValueError(f"Expected {self.dim} channels and nonempty feature dimensions")
        if not spatial_feat.is_floating_point():
            raise ValueError("Fusion inputs must be floating point tensors")

        spatial = spatial_feat.permute(0, 2, 1, 3, 4).reshape(
            batch * classes, channels, height, width
        ).contiguous()
        semantic = class_feat.permute(0, 2, 1, 3, 4).reshape_as(spatial).contiguous()
        spatial = self.spatial_proj(spatial)
        semantic = self.class_proj(semantic)
        expert_input = torch.cat((spatial, semantic), dim=1)
        affinities, paired, descriptors = self._neighborhood_statistics(spatial, semantic)
        gate_weights = self.gate_net(torch.cat((
            expert_input, descriptors.to(dtype=expert_input.dtype)
        ), dim=1))
        alpha = self.correction_max * self.correction_logits.sigmoid()

        fused = 0
        for i, expert in enumerate(self.point_experts):
            feature = expert(expert_input)
            if i == 0:
                output = feature
            elif i == 1:
                output = feature + alpha[0].to(feature.dtype) * (feature - self._mean3x3(feature))
            elif i == 2:
                output = feature + alpha[1].to(feature.dtype) * (
                    self._region_average(feature, affinities) - feature
                )
            else:
                output = feature + alpha[2].to(feature.dtype) * (
                    self._paired_direction_average(feature, paired) - feature
                )
            fused = fused + gate_weights[:, i:i + 1] * output

        fused = fused + self.res_scale * self.residual(expert_input)
        return fused.reshape(batch, classes, channels, height, width).permute(0, 2, 1, 3, 4)

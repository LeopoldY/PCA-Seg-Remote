# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License, Version 2.0.
"""RS-DINO image encoder adapted from RSKT-Seg.

The architecture and checkpoint unwrapping follow RSKT-Seg's ``BuildRSIB``
path: a DINO ViT-B/8 whose teacher/backbone prefixes are stripped before a
non-strict load.  Keeping this local makes the adapted PCA-Seg repository
self-contained and avoids a runtime dependency on the reference checkout.
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device
    )
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(
            batch, tokens, channels
        )
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_prob: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = (
            DropPath(drop_path_prob)
            if drop_path_prob > 0.0
            else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class RSDinoVisionTransformer(nn.Module):
    """DINO ViT-B/8 with RSKT-compatible parameter names and features."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 8,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        path_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_prob=path_rates[index],
                norm_layer=norm_layer,
            )
            for index in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def interpolate_pos_encoding(self, x, height: int, width: int):
        patch_count = x.shape[1] - 1
        reference_count = self.pos_embed.shape[1] - 1
        if patch_count == reference_count and height == width:
            return self.pos_embed

        class_position = self.pos_embed[:, :1]
        patch_position = self.pos_embed[:, 1:]
        reference_size = int(math.sqrt(reference_count))
        target_height = height // self.patch_embed.patch_size
        target_width = width // self.patch_embed.patch_size
        patch_position = F.interpolate(
            patch_position.reshape(
                1, reference_size, reference_size, self.embed_dim
            ).permute(0, 3, 1, 2),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
        )
        patch_position = patch_position.permute(0, 2, 3, 1).reshape(
            1, -1, self.embed_dim
        )
        return torch.cat((class_position, patch_position), dim=1)

    def prepare_tokens(self, x):
        batch, _, height, width = x.shape
        patch_tokens = self.patch_embed(x)
        class_tokens = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((class_tokens, patch_tokens), dim=1)
        tokens = tokens + self.interpolate_pos_encoding(
            tokens, height, width
        )
        return self.pos_drop(tokens)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]

    def get_intermediate_layers(self, x, n: int = 1) -> List[torch.Tensor]:
        x = self.prepare_tokens(x)
        output = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if len(self.blocks) - index <= n:
                output.append(self.norm(x))
        return output


def build_rs_dino(weights: str) -> RSDinoVisionTransformer:
    """Build and load the RSIB checkpoint exactly as RSKT-Seg expects."""
    checkpoint_path = Path(weights).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"RS-DINO pretrained weights not found: {checkpoint_path}"
        )

    model = RSDinoVisionTransformer(patch_size=8)
    # PyTorch 2.6+ defaults to ``weights_only=True``.  The trusted RSKT RSIB
    # checkpoint contains NumPy scalar metadata in addition to tensors, so it
    # must use the legacy loader.  Keep the fallback for older PyTorch builds
    # that do not expose the keyword.
    try:
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "teacher" in checkpoint:
        checkpoint = checkpoint["teacher"]
    if not isinstance(checkpoint, dict):
        raise ValueError("RS-DINO checkpoint must contain a state dictionary.")
    state_dict = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in checkpoint.items()
    }
    model.load_state_dict(state_dict, strict=False)
    return model.float()

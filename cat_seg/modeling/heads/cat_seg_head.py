# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
from einops import rearrange

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.cat_seg_predictor import CATSegPredictor


@SEM_SEG_HEADS_REGISTRY.register()
class CATSegHead(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
        clip_pretrained: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution
        self.clip_name=clip_pretrained
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": CATSegPredictor(
                cfg,
            ),
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
        }

    def forward(
        self,
        features,
        guidance_features,
        batched_inputs=None,
        dino_features=None,
        dino_guidance=None,
        prompt=None,
        gt_cls=None,
        need_loss=False,
    ):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        def to_feature_map(feature):
            if 'eva' in self.clip_name.lower():
                tokens = feature
            else:
                tokens = feature[:, 1:, :]
            return rearrange(
                tokens,
                "b (h w) c -> b c h w",
                h=self.feature_resolution[0],
                w=self.feature_resolution[1],
            )

        if isinstance(features, (list, tuple)):
            img_feat = [to_feature_map(feature) for feature in features]
        else:
            img_feat = to_feature_map(features)
        
        return self.predictor(
            x=img_feat,
            vis_guidance=guidance_features,
            dino_features=dino_features,
            dino_guidance=dino_guidance,
            batched_inputs=batched_inputs,
            prompt=prompt,
            gt_cls=gt_cls,
            need_loss=need_loss,
        )

# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import logging
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom
from cat_seg.utils.util import AverageMeter
from einops import rearrange
from cat_seg.utils.util import init_log
from cat_seg.modeling.backbone.rs_dino import build_rs_dino
from cat_seg.modeling.backbone.remote_clip import (
    build_remote_clip_visual, same_grid_content_loss, same_grid_regions,
    semantic_distribution_loss, build_remote_clip_semantic_teacher,
)
from torchvision.ops import roi_align

@META_ARCH_REGISTRY.register()
class CATSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
        rs_dino_cfg: dict,
        rs_dino_distill_cfg: dict,
        remote_clip_distill_cfg: dict,
        clip_rotation_enabled: bool,
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.Similarity = AverageMeter()
        self.logging = init_log('global', logging.INFO)
        self.logging.propagate = 0

        self.count = 0

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        self.clip_finetune = clip_finetune
      
        self.sliding_window = sliding_window
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" or clip_pretrained =="EVA02-CLIP-B-16" else (336, 336)
        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" or clip_pretrained =="EVA02-CLIP-B-16" else 1024
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)
        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" or clip_pretrained =="EVA02-CLIP-B-16" else [7, 15] 
        self.layers = []
        self.clip_name=clip_pretrained
        self.clip_rotation_enabled = bool(clip_rotation_enabled)
        self.frozen_backbone(clip_finetune)
        for l in self.layer_indexes:
            if 'eva' in self.clip_name.lower():
                self.sem_seg_head.predictor.clip_model.visual.blocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))
            else:
                self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))

        self.rs_dino_enabled = bool(rs_dino_cfg["enabled"])
        if self.rs_dino_enabled:
            self.rs_dino_resolution = tuple(rs_dino_cfg["input_size"])
            if self.rs_dino_resolution != (384, 384):
                raise ValueError(
                    "The RSKT-Seg RS-DINO path requires INPUT_SIZE [384, 384] "
                    "to align its 24x24 cost and 48x48/96x96 guidance maps "
                    "with PCA-Seg."
                )
            self.rs_dino_patch_size = 8
            self.rs_dino_model = build_rs_dino(rs_dino_cfg["weights"])
            self._set_rs_dino_finetune(rs_dino_cfg["finetune"])
            self.rs_dino_cost_projection = nn.Conv2d(
                768,
                rs_dino_cfg["text_guidance_dim"],
                kernel_size=2,
                stride=2,
            )
            decoder_guidance_dims = rs_dino_cfg["decoder_guidance_dims"]
            self.rs_dino_guidance_proj1 = nn.Conv2d(
                768, decoder_guidance_dims[0], kernel_size=1
            )
            self.rs_dino_guidance_proj2 = nn.ConvTranspose2d(
                768, decoder_guidance_dims[1], kernel_size=2, stride=2
            )

        self.rs_dino_distill_enabled = bool(
            rs_dino_distill_cfg["enabled"]
        )
        self.rs_dino_distill_weights = rs_dino_distill_cfg["weights"]
        self.rs_dino_distill_resolution = tuple(
            rs_dino_distill_cfg["input_size"]
        )
        self.rs_dino_distill_target_grid = tuple(
            rs_dino_distill_cfg["target_grid"]
        )
        self.rs_dino_distill_loss_weight = float(
            rs_dino_distill_cfg["loss_weight"]
        )
        self.rs_dino_distill_student_context = str(
            rs_dino_distill_cfg["student_context"]
        ).lower()
        if self.rs_dino_enabled and self.rs_dino_distill_enabled:
            raise ValueError(
                "RS_DINO and RS_DINO_DISTILL are mutually exclusive."
            )
        if self.rs_dino_distill_enabled:
            if 'eva' not in self.clip_name.lower() and self.clip_name not in {'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'}:
                raise ValueError(
                    "RS-DINO context distillation requires EVA CLIP or OpenAI ViT-B/16 or ViT-L/14."
                )
            if self.rs_dino_distill_student_context not in {"q", "csa"}:
                raise ValueError(
                    "RS_DINO_DISTILL.STUDENT_CONTEXT must be 'q' or 'csa'."
                )
        # Do not register the frozen teacher as a child nn.Module: it must not
        # enter optimizer/DDP/state_dict and inference must load no RSIB weight.
        object.__setattr__(self, "_rs_dino_distill_teacher", None)

        self.remote_clip_distill_enabled = bool(
            remote_clip_distill_cfg["enabled"]
        )
        self.remote_clip_distill_weights = remote_clip_distill_cfg["weights"]
        self.remote_clip_distill_model_name = remote_clip_distill_cfg[
            "model_name"
        ]
        self.remote_clip_distill_same_grid = bool(remote_clip_distill_cfg.get("same_grid", False))
        self.remote_clip_distill_objective = remote_clip_distill_cfg.get("objective", "cosine")
        self.remote_clip_distill_temperature = float(remote_clip_distill_cfg.get("temperature", 2.0))
        self.remote_clip_distill_logit_scale = float(remote_clip_distill_cfg.get("logit_scale", 10.0))
        if self.remote_clip_distill_objective not in {"cosine", "semantic_kl"}:
            raise ValueError("REMOTE_CLIP_DISTILL.OBJECTIVE must be cosine or semantic_kl.")
        if self.remote_clip_distill_enabled and self.remote_clip_distill_objective == "semantic_kl":
            import math
            if not self.remote_clip_distill_same_grid:
                raise ValueError("semantic_kl requires SAME_GRID for co-registered regions.")
            if any(not math.isfinite(x) or x <= 0 for x in (
                    self.remote_clip_distill_temperature, self.remote_clip_distill_logit_scale)):
                raise ValueError("Semantic KL temperature and scale must be positive and finite.")
        self.register_buffer("_remote_clip_text_anchors", None, persistent=False)
        self.register_buffer("_student_clip_text_anchors", None, persistent=False)
        self.remote_clip_distill_input_size = tuple(
            remote_clip_distill_cfg["input_size"]
        )
        self.remote_clip_distill_loss_weight = float(
            remote_clip_distill_cfg["loss_weight"]
        )
        self.remote_clip_distill_max_regions = int(
            remote_clip_distill_cfg["max_regions"]
        )
        self.remote_clip_distill_min_region_size = int(
            remote_clip_distill_cfg["min_region_size"]
        )
        self.remote_clip_distill_teacher_batch_size = int(
            remote_clip_distill_cfg["teacher_batch_size"]
        )
        if self.remote_clip_distill_enabled:
            if 'eva' not in self.clip_name.lower() and self.clip_name not in {'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'}:
                raise ValueError(
                    "RemoteCLIP content distillation requires EVA CLIP or OpenAI ViT-B/16 or ViT-L/14."
                )
            if self.remote_clip_distill_max_regions < 1:
                raise ValueError(
                    "REMOTE_CLIP_DISTILL.MAX_REGIONS must be at least 1."
                )
            if self.remote_clip_distill_teacher_batch_size < 1:
                raise ValueError(
                    "REMOTE_CLIP_DISTILL.TEACHER_BATCH_SIZE must be at least 1."
                )
        # As with RS-DINO, keep the frozen teacher outside the module tree so
        # DDP, the optimizer, and segmentation checkpoints only see the student.
        object.__setattr__(self, "_remote_clip_distill_teacher", None)


    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "rs_dino_cfg": {
                "enabled": cfg.MODEL.SEM_SEG_HEAD.RS_DINO.ENABLED,
                "weights": cfg.MODEL.SEM_SEG_HEAD.RS_DINO.WEIGHTS,
                "finetune": cfg.MODEL.SEM_SEG_HEAD.RS_DINO.FINETUNE,
                "input_size": cfg.MODEL.SEM_SEG_HEAD.RS_DINO.INPUT_SIZE,
                "text_guidance_dim": cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM,
                "decoder_guidance_dims": cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS,
            },
            "rs_dino_distill_cfg": {
                "enabled": cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.ENABLED,
                "weights": cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.WEIGHTS,
                "input_size": cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.INPUT_SIZE,
                "target_grid": cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.TARGET_GRID,
                "loss_weight": cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.LOSS_WEIGHT,
                "student_context": cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.STUDENT_CONTEXT,
            },
            "remote_clip_distill_cfg": {
                "enabled": cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.ENABLED,
                "weights": cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.WEIGHTS,
                "model_name": cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.MODEL_NAME,
                "input_size": cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.INPUT_SIZE,
                "same_grid": cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.SAME_GRID,
                "objective": cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.OBJECTIVE,
                "temperature": cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.TEMPERATURE,
                "logit_scale": cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.LOGIT_SCALE,
                "loss_weight": cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.LOSS_WEIGHT,
                "max_regions": cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.MAX_REGIONS,
                "min_region_size": cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.MIN_REGION_SIZE,
                "teacher_batch_size": cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.TEACHER_BATCH_SIZE,
            },
            "clip_rotation_enabled": cfg.MODEL.SEM_SEG_HEAD.CLIP_ROTATION.ENABLED,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _set_rs_dino_finetune(self, finetune):
        """Match RSKT-Seg's frozen/attention/full RS-DINO modes."""
        if finetune not in {"frozen", "attention", "full"}:
            raise ValueError(
                "RS_DINO.FINETUNE must be frozen, attention, or full."
            )
        for name, parameter in self.rs_dino_model.named_parameters():
            if finetune == "attention":
                parameter.requires_grad = (
                    "attn.qkv.weight" in name or "pos_embed" in name
                )
            elif finetune == "full":
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    def _extract_rs_dino_features(self, normalized_images):
        dino_images = F.interpolate(
            normalized_images,
            size=self.rs_dino_resolution,
            mode="bilinear",
            align_corners=False,
        )
        intermediate = self.rs_dino_model.get_intermediate_layers(
            dino_images, n=12
        )
        patch_height = dino_images.shape[-2] // self.rs_dino_patch_size
        patch_width = dino_images.shape[-1] // self.rs_dino_patch_size

        def unfold(tokens):
            return rearrange(
                tokens[:, 1:, :],
                "B (H W) C -> B C H W",
                H=patch_height,
                W=patch_width,
            )

        # RSKT-Seg uses the final layer for its text correlation and layers
        # 4/8 as the two decoder guidance scales.
        cost_feature = self.rs_dino_cost_projection(unfold(intermediate[-1]))
        guidance = [
            self.rs_dino_guidance_proj1(unfold(intermediate[3])),
            self.rs_dino_guidance_proj2(unfold(intermediate[7])),
        ]
        return cost_feature, guidance

    def _encode_clip_once(self, clip_images, distill=False):
        """Encode one orientation and isolate its forward-hook features."""
        self.layers = []
        context = None
        if 'eva' in self.clip_name.lower():
            mode = 'csa_vfm_distill' if distill else 'csa'
            encoded = self.sem_seg_head.predictor.clip_model.encode_dense(
                clip_images,
                normalize=True,
                keep_shape=False,
                mode=mode,
            )
            if distill:
                clip_features, context = encoded
            else:
                clip_features = encoded
            image_features = clip_features
        else:
            context_inputs = []
            hook = None
            if distill:
                # RSKT-Seg's forward_dense uses the final block's V path.
                # Capture its normalized input to expose pre-output-projection
                # Q/K for relation distillation without changing dense features.
                last_block = self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[-1]
                hook = last_block.ln_1.register_forward_hook(
                    lambda module, args, output: context_inputs.append(output)
                )
            try:
                clip_features = self.sem_seg_head.predictor.clip_model.encode_image(
                    clip_images, dense=True
                )
            finally:
                if hook is not None:
                    hook.remove()
            if distill:
                normalized = context_inputs[0][1:]  # remove CLS, L B C
                attention = last_block.attn
                biases = (attention.in_proj_bias.chunk(3) if attention.in_proj_bias is not None
                          else (None, None, None))
                def project_context(weight, bias):
                    projected = F.linear(normalized, weight, bias)
                    return rearrange(
                        projected, 'L B (H D) -> (B H) L D',
                        H=attention.num_heads,
                    )
                context = (
                    project_context(attention.q_proj_weight, biases[0]),
                    project_context(attention.k_proj_weight, biases[1]),
                )
            image_features = clip_features[:, 1:, :]
        return clip_features, image_features, list(self.layers), context

    def _encode_clip_directions(self, clip_images):
        """Encode 0/90/180/270 degrees with the same CLIP backbone."""
        distill = self.training and self.rs_dino_distill_enabled
        original, image_features, guidance_layers, context = (
            self._encode_clip_once(clip_images, distill=distill)
        )
        if not self.clip_rotation_enabled:
            self.layers = guidance_layers
            return original, image_features, guidance_layers, context

        directional_features = [original]
        for quarter_turns in (1, 2, 3):
            rotated_images = torch.rot90(
                clip_images, k=quarter_turns, dims=(-2, -1)
            )
            rotated, _, _, _ = self._encode_clip_once(rotated_images)
            directional_features.append(rotated)
        # Only the original direction supplies multi-scale decoder guidance.
        self.layers = guidance_layers
        return (
            directional_features,
            image_features,
            guidance_layers,
            context,
        )

    def _build_clip_guidance(self, image_features, guidance_layers):
        res3 = rearrange(
            image_features, "B (H W) C -> B C H W", H=24
        )
        if len(guidance_layers) < 2:
            raise RuntimeError(
                "CLIP intermediate feature hooks did not return two layers."
            )
        if 'eva' in self.clip_name.lower():
            res4 = rearrange(
                guidance_layers[0][:, 1:, :],
                "B (H W) C -> B C H W",
                H=24,
            )
            res5 = rearrange(
                guidance_layers[1][:, 1:, :],
                "B (H W) C -> B C H W",
                H=24,
            )
        else:
            res4 = rearrange(
                guidance_layers[0][1:, :, :],
                "(H W) B C -> B C H W",
                H=24,
            )
            res5 = rearrange(
                guidance_layers[1][1:, :, :],
                "(H W) B C -> B C H W",
                H=24,
            )
        return {
            'res5': self.upsample2(res5),
            'res4': self.upsample1(res4),
            'res3': res3,
        }

    def _get_rs_dino_distill_teacher(self):
        teacher = self._rs_dino_distill_teacher
        if teacher is None:
            teacher = build_rs_dino(self.rs_dino_distill_weights)
            teacher.requires_grad_(False)
            teacher.eval()
            teacher.to(self.device)
            object.__setattr__(self, "_rs_dino_distill_teacher", teacher)
        else:
            teacher.eval()
        return teacher

    @staticmethod
    def _context_similarity(tokens):
        tokens = F.normalize(tokens.float(), dim=-1)
        return torch.einsum('bnc,bmc->bnm', tokens, tokens)

    def _student_context_similarity(self, context, batch_size):
        q_feature, k_feature = context

        def merge_heads(feature):
            if feature.shape[0] % batch_size != 0:
                raise ValueError("Invalid EVA context batch/head layout.")
            num_heads = feature.shape[0] // batch_size
            feature = rearrange(
                feature,
                '(B H) N C -> B N (H C)',
                B=batch_size,
                H=num_heads,
            )
            source_grid = int(feature.shape[1] ** 0.5)
            if source_grid * source_grid != feature.shape[1]:
                raise ValueError("EVA context tokens must form a square grid.")
            if (source_grid, source_grid) != self.rs_dino_distill_target_grid:
                feature = rearrange(
                    feature,
                    'B (H W) C -> B C H W',
                    H=source_grid,
                    W=source_grid,
                )
                feature = F.interpolate(
                    feature,
                    size=self.rs_dino_distill_target_grid,
                    mode='bilinear',
                    align_corners=False,
                )
                feature = rearrange(feature, 'B C H W -> B (H W) C')
            return self._context_similarity(feature)

        q_similarity = merge_heads(q_feature)
        if self.rs_dino_distill_student_context == "q":
            return q_similarity
        return (q_similarity + merge_heads(k_feature)) / 2.0

    def _rs_dino_context_distillation_loss(self, images, context):
        if context is None:
            raise RuntimeError("EVA CLIP did not return distillation context.")
        teacher_images = [
            (image - self.pixel_mean) / self.pixel_std for image in images
        ]
        teacher_images = ImageList.from_tensors(
            teacher_images, self.size_divisibility
        ).tensor
        teacher_images = F.interpolate(
            teacher_images,
            size=self.rs_dino_distill_resolution,
            mode='bilinear',
            align_corners=False,
        )
        teacher = self._get_rs_dino_distill_teacher()
        with torch.no_grad():
            teacher_tokens = teacher.get_intermediate_layers(
                teacher_images, n=1
            )[-1][:, 1:, :]
            teacher_grid = int(teacher_tokens.shape[1] ** 0.5)
            teacher_features = rearrange(
                teacher_tokens,
                'B (H W) C -> B C H W',
                H=teacher_grid,
                W=teacher_grid,
            )
            teacher_features = F.adaptive_avg_pool2d(
                teacher_features, self.rs_dino_distill_target_grid
            )
            teacher_tokens = rearrange(
                teacher_features, 'B C H W -> B (H W) C'
            )
            teacher_similarity = self._context_similarity(teacher_tokens)
        student_similarity = self._student_context_similarity(
            context, len(images)
        )
        return (teacher_similarity - student_similarity).norm(
            p=2, dim=-1
        ).mean()

    def _get_remote_clip_distill_teacher(self):
        teacher = self._remote_clip_distill_teacher
        if teacher is None:
            if self.remote_clip_distill_objective == "semantic_kl":
                predictor = self.sem_seg_head.predictor
                teacher, anchors = build_remote_clip_semantic_teacher(
                    self.remote_clip_distill_model_name, self.remote_clip_distill_weights,
                    predictor.class_texts, predictor.prompt_templates, self.device)
                self._remote_clip_text_anchors = anchors
                # Cached unadapted OpenAI class embeddings: [concept, prompt, dim].
                # Do not use attribute-enriched text as a moving semantic KD target.
                self._student_clip_text_anchors = F.normalize(
                    predictor.text_features.detach().float().mean(dim=1), dim=-1).to(self.device)
            else:
                teacher = build_remote_clip_visual(
                    self.remote_clip_distill_model_name,
                    self.remote_clip_distill_weights,
                )
            teacher.to(self.device)
            teacher.requires_grad_(False)
            teacher.eval()
            object.__setattr__(self, "_remote_clip_distill_teacher", teacher)
        else:
            teacher.eval()
        return teacher

    def _semantic_region_boxes(self, batched_inputs):
        """Build DeCLIP crop proposals from the semantic training targets."""
        regions = []
        for batch_index, sample in enumerate(batched_inputs):
            target = sample["sem_seg"].to(self.device)
            height, width = target.shape[-2:]
            candidates = []
            for class_id in torch.unique(target).tolist():
                if class_id == self.sem_seg_head.ignore_value:
                    continue
                ys, xs = torch.where(target == class_id)
                if xs.numel() == 0:
                    continue
                x1, x2 = xs.min(), xs.max() + 1
                y1, y2 = ys.min(), ys.max() + 1
                if (
                    (x2 - x1).item() < self.remote_clip_distill_min_region_size
                    or (y2 - y1).item()
                    < self.remote_clip_distill_min_region_size
                ):
                    continue
                candidates.append((xs.numel(), x1, y1, x2, y2))

            candidates.sort(key=lambda item: item[0], reverse=True)
            for _, x1, y1, x2, y2 in candidates[
                : self.remote_clip_distill_max_regions
            ]:
                regions.append(
                    torch.stack(
                        [
                            target.new_tensor(batch_index),
                            x1,
                            y1,
                            x2,
                            y2,
                        ]
                    ).float()
                )

            # Degenerate smoke samples can contain only ignore/tiny regions.
            # A whole-image crop keeps the content objective well-defined.
            if not candidates:
                regions.append(
                    target.new_tensor(
                        [batch_index, 0, 0, width, height],
                        dtype=torch.float32,
                    )
                )
        return torch.stack(regions)

    def _remote_clip_content_distillation_loss(
        self, normalized_images, student_dense_features, batched_inputs
    ):
        """DeCLIP content loss with RemoteCLIP as the frozen crop teacher."""
        boxes = self._semantic_region_boxes(batched_inputs)
        if self.remote_clip_distill_objective == "semantic_kl":
            teacher = self._get_remote_clip_distill_teacher()
            student_regions, teacher_regions = same_grid_regions(
                teacher, normalized_images, student_dense_features, boxes,
                self.remote_clip_distill_input_size, self.remote_clip_distill_teacher_batch_size)
            return semantic_distribution_loss(
                student_regions, teacher_regions, self._student_clip_text_anchors,
                self._remote_clip_text_anchors, self.remote_clip_distill_temperature,
                self.remote_clip_distill_logit_scale)
        if self.remote_clip_distill_same_grid:
            return same_grid_content_loss(
                self._get_remote_clip_distill_teacher(), normalized_images,
                student_dense_features, boxes, self.remote_clip_distill_input_size,
                self.remote_clip_distill_teacher_batch_size,
            )
        teacher_crops = roi_align(
            normalized_images,
            boxes,
            output_size=self.remote_clip_distill_input_size,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )

        teacher = self._get_remote_clip_distill_teacher()
        teacher_features = []
        with torch.no_grad():
            for crops in teacher_crops.split(
                self.remote_clip_distill_teacher_batch_size
            ):
                teacher_features.append(
                    F.normalize(teacher(crops).float(), dim=-1)
                )
        teacher_features = torch.cat(teacher_features)

        batch_size, num_tokens, channels = student_dense_features.shape
        grid_size = int(num_tokens ** 0.5)
        if grid_size * grid_size != num_tokens:
            raise ValueError("Student dense features must form a square grid.")
        student_map = rearrange(
            student_dense_features,
            "B (H W) C -> B C H W",
            H=grid_size,
            W=grid_size,
        )
        student_boxes = boxes.clone()
        student_boxes[:, [1, 3]] *= grid_size / normalized_images.shape[-1]
        student_boxes[:, [2, 4]] *= grid_size / normalized_images.shape[-2]
        student_features = roi_align(
            student_map,
            student_boxes,
            output_size=(1, 1),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        ).flatten(1)
        student_features = F.normalize(student_features.float(), dim=-1)
        if student_features.shape[-1] != teacher_features.shape[-1]:
            raise ValueError(
                "Student and RemoteCLIP content dimensions differ: "
                f"{student_features.shape[-1]} vs {teacher_features.shape[-1]}."
            )
        return 1.0 - (student_features * teacher_features).sum(-1).mean()
    
   

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)
        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False)
        if self.rs_dino_enabled:
            dino_features, dino_guidance = self._extract_rs_dino_features(
                clip_images.tensor
            )
        else:
            dino_features, dino_guidance = None, None
        (
            clip_features,
            image_features,
            guidance_layers,
            student_context,
        ) = self._encode_clip_directions(clip_images_resized)
        features = self._build_clip_guidance(
            image_features, guidance_layers
        )
        # outputs = self.sem_seg_head(clip_features, features)
        # self.Similarity.update(similarity)
        # self.count = self.count + 1
        # if self.count % 200 == 0 :
        #     self.logging.info("当前的相似度值为%s"%(self.Similarity.avg))

        if self.training:
            outputs,loss_organ, similarity = self.sem_seg_head(
                clip_features,
                features,
                batched_inputs=batched_inputs,
                dino_features=dino_features,
                dino_guidance=dino_guidance,
                need_loss=True)
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value
            outputs = outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss, "loss_organ": loss_organ}
            if self.rs_dino_distill_enabled:
                loss_context = self._rs_dino_context_distillation_loss(
                    images, student_context
                )
                losses["loss_rs_dino_context"] = (
                    self.rs_dino_distill_loss_weight * loss_context
                )
            if self.remote_clip_distill_enabled:
                loss_content = self._remote_clip_content_distillation_loss(
                    clip_images.tensor,
                    image_features,
                    batched_inputs,
                )
                remote_loss_name = ("loss_remote_clip_semantic" if
                                    self.remote_clip_distill_objective == "semantic_kl"
                                    else "loss_remote_clip_content")
                losses[remote_loss_name] = (
                    self.remote_clip_distill_loss_weight * loss_content
                )
                if self.remote_clip_distill_objective == "semantic_kl":
                    from detectron2.utils.events import get_event_storage
                    from detectron2.utils import comm
                    raw_metric = comm.reduce_dict({"kl": loss_content.detach()})["kl"]
                    # Diagnostics are not entries in the optimizer's loss dict.
                    try:
                        storage = get_event_storage()
                    except AssertionError:
                        storage = None
                    if storage is not None:
                        storage.put_scalar("remote_clip_semantic_kl_raw", raw_metric.item())
            return losses
        else:
            outputs = self.sem_seg_head(
                clip_features,
                features,
                batched_inputs,
                dino_features=dino_features,
                dino_guidance=dino_guidance,
            )
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])
            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results

    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)
        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)
        images = (image - self.pixel_mean) / self.pixel_std
        clip_images_normalized = (
            image - self.clip_pixel_mean
        ) / self.clip_pixel_std
        # The distillation teacher is deliberately absent from inference.
        if self.rs_dino_enabled:
            dino_features, dino_guidance = self._extract_rs_dino_features(
                clip_images_normalized
            )
        else:
            dino_features, dino_guidance = None, None
        clip_images = F.interpolate(
            clip_images_normalized,
            size=self.clip_resolution,
            mode='bilinear',
            align_corners=False,
        )
        (
            clip_features,
            image_features,
            guidance_layers,
            _,
        ) = self._encode_clip_directions(clip_images)
        features = self._build_clip_guidance(
            image_features, guidance_layers
        )
        outputs = self.sem_seg_head(
            clip_features,
            features,
            batched_inputs,
            dino_features=dino_features,
            dino_guidance=dino_guidance,
        )  # torch.Size([5, 59, 96, 96])
        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.
        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]
    
    def frozen_backbone(self,clip_finetune):
        if 'eva' in self.clip_name.lower():
            for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
                if 'visual.blocks' in name:
                    if "attn" in name:
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    else:
                        params.requires_grad = False
                elif 'text.transformer' in name:
                    if 'resblocks' in name:
                        params.requires_grad = True if "in_proj_weight" in name else False
                    else:
                        params.requires_grad = False
                else:
                    params.requires_grad = False
        else:
            for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
                if "transformer" in name:
                    if clip_finetune == "prompt":
                        params.requires_grad = True if "prompt" in name else False
                    elif clip_finetune == "attention":
                        if "attn" in name:
                            # QV fine-tuning for attention blocks
                            params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                        elif "position" in name:
                            params.requires_grad = True
                        else:
                            params.requires_grad = False
                    elif clip_finetune == "full":
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                else:
                    params.requires_grad = False

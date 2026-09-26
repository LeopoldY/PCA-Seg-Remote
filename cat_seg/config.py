# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_cat_seg_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"

    cfg.DATASETS.VAL_ALL = ("coco_2017_val_all_stuff_sem_seg",)

    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    # One optimizer update per dataloader batch unless explicitly overridden.
    cfg.SOLVER.GRAD_ACCUM_STEPS = 1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    # zero shot config
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = "datasets/ADE20K_2021_17_01/ADE20K_847.json"
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = "datasets/ADE20K_2021_17_01/ADE20K_847.json"
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES = "datasets/coco/coco_stuff/split/seen_indexes.json"
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES = "datasets/coco/coco_stuff/split/unseen_indexes.json"

    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED = "ViT-B/16"

    cfg.MODEL.PROMPT_ENSEMBLE = False
    cfg.MODEL.PROMPT_ENSEMBLE_TYPE = "single"

    cfg.MODEL.CLIP_PIXEL_MEAN = [122.7709383, 116.7460125, 104.09373615]
    cfg.MODEL.CLIP_PIXEL_STD = [68.5005327, 66.6321579, 70.3231630]
    # three styles for clip classification, crop, mask, cropmask

    cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM = 128
    cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM = 128

    cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS = [64, 32]
    cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS = [256, 128]
    cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS = [32, 16]

    cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS = 4
    cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS = 4
    cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS = 128
    cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES = [6, 6]
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION = [24, 24]
    cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES = 12
    cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE = "linear"

    cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH = 0
    cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH = 0

    # Optional ExCEL Text Semantic Enrichment (TSE) for cost construction.
    cfg.MODEL.SEM_SEG_HEAD.ATTR_FUSION = CN()
    cfg.MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED = False
    cfg.MODEL.SEM_SEG_HEAD.ATTR_FUSION.DATABASE_PATH = ""
    cfg.MODEL.SEM_SEG_HEAD.ATTR_FUSION.NUM_CLUSTERS = 112
    # ExCEL's topK argument keeps the highest 90% of cluster logits.
    cfg.MODEL.SEM_SEG_HEAD.ATTR_FUSION.TOP_K = 0.9

    # Optional replacement for DualFeatureMoE.  The original module remains
    # the default so existing YAML files and checkpoints keep the same module
    # hierarchy and state-dict keys.
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_FUSION = CN()
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_FUSION.TYPE = "dual_feature_moe"
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_FUSION.MULTISCALE_MOE = CN()
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_FUSION.MULTISCALE_MOE.KERNEL_SIZES = [1, 3, 5, 7]
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_FUSION.RS_STRUCTURE_MOE = CN()
    rs_structure = cfg.MODEL.SEM_SEG_HEAD.FEATURE_FUSION.RS_STRUCTURE_MOE
    rs_structure.TAU = 0.1
    rs_structure.CORRECTION_MAX = 0.5
    rs_structure.CORRECTION_INIT = 0.1
    rs_structure.DETACH_GUIDANCE = True
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_FUSION.SPARSE_TRANSFORMER = CN()
    sparse_fusion = (
        cfg.MODEL.SEM_SEG_HEAD.FEATURE_FUSION.SPARSE_TRANSFORMER
    )
    sparse_fusion.NUM_HEADS = 4
    sparse_fusion.NUM_ROUTED_EXPERTS = 4
    sparse_fusion.TOP_K = 1
    sparse_fusion.EXPERT_RATIO = 1.0
    sparse_fusion.ATTN_DROPOUT = 0.0
    sparse_fusion.ROUTER_BALANCE_WEIGHT = 0.01

    # Optional RS-DINO branch adapted from RSKT-Seg.  It is opt-in so all
    # existing PCA-Seg configurations preserve their original behavior.
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO = CN()
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO.ENABLED = False
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO.WEIGHTS = ""
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO.FINETUNE = "frozen"
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO.INPUT_SIZE = [384, 384]

    # RSKT-Seg-style four-direction encoding with one shared CLIP backbone.
    # The rotated feature maps are aligned back to the original orientation
    # before their cost volumes are concatenated along the prompt axis.
    cfg.MODEL.SEM_SEG_HEAD.CLIP_ROTATION = CN()
    cfg.MODEL.SEM_SEG_HEAD.CLIP_ROTATION.ENABLED = False
    cfg.MODEL.SEM_SEG_HEAD.CLIP_ROTATION.NUM_DIRECTIONS = 4

    # DeCLIP context distillation.  Unlike RS_DINO above, this path is a
    # training-only frozen teacher: it is loaded lazily, is not registered in
    # the model state_dict, and is never needed for inference.
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL = CN()
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.ENABLED = False
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.WEIGHTS = ""
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.INPUT_SIZE = [384, 384]
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.TARGET_GRID = [24, 24]
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.LOSS_WEIGHT = 1.0e-5
    cfg.MODEL.SEM_SEG_HEAD.RS_DINO_DISTILL.STUDENT_CONTEXT = "q"

    # The content half of DeCLIP-style dual-teacher distillation. RS-DINO
    # supervises spatial context above, while frozen RemoteCLIP crop features
    # supervise the student's dense region features. Both teachers are
    # training-only and are excluded from checkpoints and inference.
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL = CN()
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.ENABLED = False
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.WEIGHTS = ""
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.MODEL_NAME = "ViT-B-32"
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.INPUT_SIZE = [224, 224]
    # Opt in to native-grid region distillation; legacy runs retain crop CLS loss.
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.SAME_GRID = False
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.OBJECTIVE = "cosine"
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.TEMPERATURE = 2.0
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.LOGIT_SCALE = 10.0
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.LOSS_WEIGHT = 5.0e-5
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.MAX_REGIONS = 8
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.MIN_REGION_SIZE = 8
    cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.TEACHER_BATCH_SIZE = 32

    cfg.SOLVER.CLIP_MULTIPLIER = 0.01

    cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE = "attention"
    cfg.MODEL.SEM_SEG_HEAD.CACHE_DIR = None
    cfg.TEST.SLIDING_WINDOW = False

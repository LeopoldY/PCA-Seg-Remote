from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from cat_seg import add_cat_seg_config
from cat_seg.modeling.attribute_fusion import load_excel_attribute_database
from cat_seg.modeling.transformer.model import build_feature_fusion, DualFeatureMoE
for dataset in ['dlrsd', 'isaid']:
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(f'configs/clip_vitb_384_{dataset}_attr20_dual_teacher_dual_feature_moe.yaml')
    head = cfg.MODEL.SEM_SEG_HEAD
    assert head.CLIP_PRETRAINED == 'ViT-B/16'
    assert head.FEATURE_FUSION.TYPE == 'dual_feature_moe'
    assert type(build_feature_fusion(128, {'type': head.FEATURE_FUSION.TYPE})) is DualFeatureMoE
    assert head.RS_DINO_DISTILL.ENABLED and head.REMOTE_CLIP_DISTILL.ENABLED
    assert not head.RS_DINO.ENABLED
    assert head.RS_DINO_DISTILL.LOSS_WEIGHT == .1 and head.REMOTE_CLIP_DISTILL.LOSS_WEIGHT == 1.
    assert head.ATTR_FUSION.NUM_CLUSTERS == 20
    load_excel_attribute_database(head.ATTR_FUSION.DATABASE_PATH, expected_dim=512, expected_num_clusters=20)
    assert cfg.SOLVER.IMS_PER_BATCH == 8 and cfg.SOLVER.MAX_ITER == 60000
    assert cfg.SOLVER.BASE_LR == .0002
    print(dataset, 'PASS: CLIP ViT-B/16, DualFeatureMoE, dual teachers, 512x20 bank, batch=8, iterations=60000')

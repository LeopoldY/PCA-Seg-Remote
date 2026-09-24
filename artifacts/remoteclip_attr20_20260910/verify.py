import hashlib
import json
from pathlib import Path
import torch
from cat_seg.modeling.attribute_fusion import load_excel_attribute_database

root = Path.cwd()
results = []
for tag, dim, model in [('b32', 512, 'ViT-B-32'), ('l14', 768, 'ViT-L-14')]:
    checkpoint = root / 'pretrained' / f'RemoteCLIP-{model}.pt'
    for dataset in ['DLRSD', 'iSAID']:
        descriptors = root / 'attributes_text' / f'{dataset}_train_descriptors.json'
        descriptions = json.loads(descriptors.read_text())
        path = root / 'attributes_text' / 'rskt_seg' / f'{dataset}_train_desc_remoteclip_{tag}_cluster_20_embedding_bank.pth'
        bank, flags = load_excel_attribute_database(str(path), expected_dim=dim, expected_num_clusters=20)
        assert flags.shape == (len(descriptions), 20)
        assert torch.isfinite(bank).all() and torch.isfinite(flags).all()
        assert (bank.norm(dim=0) > 0).all()
        assert ((flags == 0) | (flags == 1)).all()
        assert (flags.sum(dim=0) > 0).all() and (flags.sum(dim=1) > 0).all()
        result = dict(dataset=dataset, encoder=f'RemoteCLIP-{model}', checkpoint=str(checkpoint.resolve()),
                      descriptors_sha256=hashlib.sha256(descriptors.read_bytes()).hexdigest(),
                      descriptor_count=sum(map(len, descriptions.values())), classes=list(descriptions),
                      bank=str(path.relative_to(root)), bank_shape=list(bank.shape), flags_shape=list(flags.shape),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(), finite=True)
        results.append(result)
        print(dataset, model, tuple(bank.shape), tuple(flags.shape), 'PASS')
(root / 'artifacts/remoteclip_attr20_20260910/verification.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from cat_seg import add_cat_seg_config
for config in sorted((root / 'configs').glob('clip_vit*_attr20_dual_teacher_aff_residual_moe.yaml')):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(str(config))
    attr = cfg.MODEL.SEM_SEG_HEAD.ATTR_FUSION
    assert attr.NUM_CLUSTERS == 20
    load_excel_attribute_database(attr.DATABASE_PATH, expected_dim=cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM, expected_num_clusters=attr.NUM_CLUSTERS)
    print(config.name, 'merged config and bank dimensions PASS')

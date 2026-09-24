"""Apply small, asserted compatibility/correctness fixes to upstream 59c90eaa."""
from pathlib import Path

def replace(path, old, new):
    p=Path(path)
    s=p.read_text()
    assert s.count(old)==1, (path, old)
    p.write_text(s.replace(old,new))

replace("train_net.py", 'os.environ["DETECTRON2_DATASETS"] = "/data1/yjj/datasets/DETECTRON2_DATASETS"',
        'os.environ.setdefault("DETECTRON2_DATASETS", "datasets")')
replace("cat_seg/data/datasets/__init__.py", '    register_coco_stuff,', '    register_remote_sensing,\n    register_coco_stuff,')
# CCA is used only in the optional, uncalled diagnostic function.
replace("cat_seg/modeling/transformer/model.py", 'from cca_zoo.models import CCA\n', '')
replace("cat_seg/modeling/transformer/model.py", 'def cca(feature_list):',
        'def cca(feature_list):\n    from cca_zoo.models import CCA')
replace("cat_seg/modeling/transformer/model.py",
        'layer(corr_embed, projected_guidance, projected_text_guidance,need_loss)',
        'layer(corr_embed, projected_guidance, projected_text_guidance, batched_inputs, need_loss=need_loss)')
# Recompute validation text embeddings after further fine-tuning.
replace("cat_seg/modeling/transformer/cat_seg_predictor.py",
        '        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]',
        '        if self.training:\n            self.cache = None\n        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]')

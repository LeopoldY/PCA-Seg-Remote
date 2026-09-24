import json
import torch
from torch import nn
from cat_seg.modeling.backbone.remote_clip import same_grid_content_loss

# Coordinate-sensitive map: matching subregions must have zero cosine loss.
class Teacher(nn.Module):
    def __init__(self, feature):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 32, stride=32)
        self.register_buffer('feature', feature)
    def encode_dense(self, images, keep_shape, mode):
        assert keep_shape and mode == 'maskclip'
        return self.feature[:len(images)]
feature=torch.randn(1, 4, 24, 24)
t=Teacher(feature)
s=feature.flatten(2).transpose(1,2).clone().requires_grad_()
boxes=torch.tensor([[0,0,0,384,384],[0,32,64,192,320]],dtype=torch.float32)
images=torch.randn(1,3,384,384)
loss=same_grid_content_loss(t,images,s,boxes,(768,768),1)
assert abs(loss.item()) < 1e-6
loss.backward(); assert s.grad is not None and all(p.grad is None for p in t.parameters())
for size in [(224,224),(384,384)]:
    try: same_grid_content_loss(t,images,s,boxes,size,1)
    except ValueError: pass
    else: raise AssertionError('Mismatch accepted')
print('PASS: identical subregion coordinates, student gradients, no teacher gradients, grid rejection')

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from cat_seg import add_cat_seg_config
cfg=get_cfg();add_deeplab_config(cfg);add_cat_seg_config(cfg)
cfg.merge_from_file('configs/clip_vitb_384_isaid_attr64_dual_teacher_dual_feature_moe_rs.yaml')
assert cfg.MODEL.SEM_SEG_HEAD.REMOTE_CLIP_DISTILL.SAME_GRID
model=build_model(cfg); model.eval()
teacher=model._get_remote_clip_distill_teacher()
images=images.cuda(); boxes=boxes.cuda()
with torch.no_grad():
    _, actual_student, _, _ = model._encode_clip_once(images)
    teacher_map=teacher.encode_dense(torch.nn.functional.interpolate(images,(768,768),mode='bilinear',align_corners=False),keep_shape=True,mode='maskclip')
assert actual_student.shape==(1,576,512)
assert teacher_map.shape==(1,512,24,24)
student_leaf=actual_student.detach().requires_grad_()
loss=same_grid_content_loss(teacher,images,student_leaf,boxes,(768,768),1)
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(student_leaf.grad).all() and student_leaf.grad.abs().sum()>0
assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
result={'student_tokens':list(actual_student.shape),'teacher_map':list(teacher_map.shape),'loss':loss.item(),'student_gradient_finite_nonzero':True,'teacher_frozen':True,'optimizer_steps':0}
print(json.dumps(result))

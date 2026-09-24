from pathlib import Path
p=Path('cat_seg/cat_seg_model.py')
s=p.read_text()
a='self.sem_seg_head(clip_features, features,need_loss=True)'
assert s.count(a)==1
p.write_text(s.replace(a,'self.sem_seg_head(clip_features, features, batched_inputs, need_loss=True)'))

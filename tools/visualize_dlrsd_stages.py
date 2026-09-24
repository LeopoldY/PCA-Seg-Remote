#!/usr/bin/env python3
"""Export real dense features, all aggregation branches, and segmentation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from cat_seg import add_cat_seg_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.checkpoint
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    model = predictor.model
    assert not model.sliding_window, 'This exporter expects whole-image inference'
    classes = model.sem_seg_head.predictor.test_class_texts
    assert len(classes) == 17
    bgr = cv2.imread(args.image)
    assert bgr is not None
    rgb = bgr[:, :, ::-1].copy()
    height, width = rgb.shape[:2]
    Image.fromarray(rgb).save(out / 'input.png')
    captured = {}
    hooks = []
    def capture(name):
        def hook(module, inputs, result):
            captured[name] = result.detach().float().cpu()
        return hook
    def capture_dense(module, inputs):
        features = inputs[0]
        features = features if isinstance(features, (list, tuple)) else [features]
        for i, feature in enumerate(features):
            captured[f'clip_direction_{i}'] = feature.detach().float().cpu()
    hooks.append(model.sem_seg_head.register_forward_pre_hook(capture_dense))
    agg = model.sem_seg_head.predictor.transformer
    for i, layer in enumerate(agg.layers):
        for branch, attr in [('spatial_first', 'swin_block_first'),
                             ('class_first', 'attention_first'),
                             ('spatial_second', 'swin_block_second'),
                             ('class_second', 'attention_second')]:
            hooks.append(getattr(layer, attr).register_forward_hook(capture(f'layer{i+1}_{branch}')))
    with torch.inference_mode():
        scores = predictor(bgr)['sem_seg'].float().cpu()
    for hook in hooks:
        hook.remove()
    assert scores.shape == (17, height, width)
    pred = scores.argmax(0).numpy().astype('uint8')
    Image.fromarray(pred).save(out / 'segmentation_ids.png')
    # Teachers use precisely the normalization and image sizes in training.
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().to(model.device)[None]
    with torch.inference_mode():
        dino_input = F.interpolate((tensor - model.pixel_mean) / model.pixel_std,
                                  size=model.rs_dino_distill_resolution, mode='bilinear', align_corners=False)
        dino = model._get_rs_dino_distill_teacher()
        captured['rs_dino_dense'] = dino.get_intermediate_layers(dino_input, n=1)[-1][:, 1:].float().cpu()
        remote_input = F.interpolate((tensor - model.clip_pixel_mean) / model.clip_pixel_std,
                                    size=model.remote_clip_distill_input_size, mode='bilinear', align_corners=False)
        remote = model._get_remote_clip_distill_teacher()
        # Standard visual forward, retaining actual final transformer patch tokens.
        remote.output_tokens = True
        _, tokens = remote(remote_input)
        captured['remoteclip_dense'] = (remote.ln_post(tokens) @ remote.proj).float().cpu()
        remote.output_tokens = False
    arrays = {k: v.numpy() for k, v in captured.items()}
    arrays['segmentation_scores'] = scores.numpy()
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez_compressed(out / 'raw_features.npz', **arrays)
    manifest = {'checkpoint': str(Path(args.checkpoint).resolve()), 'image': args.image,
                'config': args.config, 'classes': classes, 'shapes': {k: list(v.shape) for k,v in arrays.items()},
                'teacher_weights': {'rs_dino': model.rs_dino_distill_weights, 'remoteclip': model.remote_clip_distill_weights},
                'dense_method': 'PC1 of per-patch L2-normalized features, centered across patches; sign fixed by largest loading. Independent 2-98 percentile display scaling. PCA colors not comparable between models.',
                'aggregation_method': 'RMS across hidden channels for each class; common min-max scale across all spatial/class branches and all 17 classes. Not class probabilities.',
                'remoteclip_method': 'Standard frozen visual forward at 224x224; final patch tokens after ln_post and projection, excluding CLS. Native 7x7 grid.',
                'inference': 'Unmodified DefaultPredictor with saved training config, including 4 rotation directions; whole-image inference.',
                'predicted_pixels': {name: int((pred == i).sum()) for i,name in enumerate(classes)}}
    def upscale(a):
        return cv2.resize(a, (width, height), interpolation=cv2.INTER_LINEAR)
    def save_heat(name, heat):
        color = (plt.get_cmap('turbo')(np.clip(upscale(heat),0,1))[:,:,:3] * 255).astype('uint8')
        Image.fromarray(color).save(out / f'{name}_heatmap.png')
        overlay = (0.5 * rgb + 0.5 * color).astype('uint8')
        Image.fromarray(overlay).save(out / f'{name}_overlay.png')
        return overlay
    dense_overlays = []
    for name, key in [('CLIP', 'clip_direction_0'), ('RS-DINO','rs_dino_dense'), ('RemoteCLIP','remoteclip_dense')]:
        x = captured[key][0]
        grid = int(x.shape[0] ** 0.5)
        assert grid * grid == x.shape[0]
        x = F.normalize(x, dim=-1)
        x = x - x.mean(0, keepdim=True)
        u,s,v = torch.linalg.svd(x, full_matrices=False)
        components = u[:,:3] * s[:3]
        for j in range(3):
            if v[j, v[j].abs().argmax()] < 0:
                components[:,j] *= -1
        pca = components.numpy().reshape(grid,grid,3)
        lo,hi = np.percentile(pca, [2,98], axis=(0,1))
        normalized = np.clip((pca-lo)/(hi-lo+1e-12),0,1)
        slug = name.lower().replace('-','_')
        dense_overlays.append(save_heat(slug, normalized[:,:,0]))
        Image.fromarray((upscale(normalized)*255).astype('uint8')).save(out / f'{slug}_pca_rgb.png')
        manifest.setdefault('dense_pca',{})[name] = {'native_grid': grid, 'explained_variance_pc1': float(s[0]**2/(s**2).sum())}
    energies = {k: (v[0].square().mean(0).sqrt().numpy()) for k,v in captured.items() if k.startswith('layer')}
    lo = min(float(x.min()) for x in energies.values())
    hi = max(float(x.max()) for x in energies.values())
    manifest['aggregation_display_range'] = [lo, hi]
    def panel(name, values, title, vmin, vmax):
        fig,axes = plt.subplots(4,5,figsize=(15,12))
        for i,ax in enumerate(axes.flat):
            ax.axis('off')
            if i < 17:
                ax.imshow(rgb)
                im = ax.imshow(upscale(values[i]), cmap='turbo', alpha=.6, vmin=vmin, vmax=vmax)
                ax.set_title(f'{i:02d}  {classes[i]}', fontsize=11)
        fig.suptitle(title, fontsize=17)
        fig.subplots_adjust(left=.02,right=.9,bottom=.02,top=.93,wspace=.08,hspace=.15)
        fig.colorbar(im,cax=fig.add_axes([.93,.15,.015,.65]),label='Channel RMS' if 'RMS' in title else 'Sigmoid score')
        fig.savefig(out / f'{name}.png',dpi=160)
        plt.close(fig)
    for key, energy in energies.items():
        folder = out / key
        folder.mkdir(exist_ok=True)
        for i, cls in enumerate(classes):
            save_heat(f'{key}/{i:02d}_{cls.replace(" ","_")}', (energy[i]-lo)/(hi-lo))
        panel(key + '_all_classes', energy, f'{key.replace("_", " ")} | All 17 classes | Channel RMS',lo,hi)
    panel('final_scores_all_classes', scores.numpy(), 'Final segmentation scores | All 17 classes',0,1)
    palette = (plt.get_cmap('tab20')(np.arange(17))[:,:3]*255).astype('uint8')
    segmentation = palette[pred]
    Image.fromarray(segmentation).save(out/'segmentation_color.png')
    seg_overlay = (rgb*.45 + segmentation*.55).astype('uint8')
    Image.fromarray(seg_overlay).save(out/'segmentation_overlay.png')
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(seg_overlay)
    ax.axis('off')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=palette[i]/255, label=name) for i,name in enumerate(classes)],
              loc='center left',bbox_to_anchor=(1, .5),frameon=False)
    fig.tight_layout()
    fig.savefig(out/'segmentation_legend.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(20,4.5))
    for ax,im,title in zip(axes,[rgb]+dense_overlays+[seg_overlay],
                           ['Input','CLIP dense / PC1','RS-DINO dense / PC1','RemoteCLIP dense / PC1','Final segmentation']):
        ax.imshow(im); ax.set_title(title); ax.axis('off')
    fig.tight_layout()
    fig.savefig(out/'overview.png',dpi=180)
    plt.close(fig)
    with open(args.checkpoint,'rb') as stream:
        digest=hashlib.sha256()
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    manifest['checkpoint_sha256']=digest.hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'README.md').write_text('''# Airplane78 特征与分割可视化

- `overview.png`：输入、三个 dense 特征 PC1 热力图、最终分割。
- `layer2_spatial_second_all_classes.png`：最后一层第二次空间聚合，全部 17 类。
- `layer2_class_second_all_classes.png`：最后一层第二次类别聚合，全部 17 类。
- 所有 `layer*` 子目录：各层、各分支、每个类别的纯热力图与叠加图。
- `segmentation_legend.png`：带类别图例的最终分割；`segmentation_ids.png` 为原始类别 ID（0–16）。
- `final_scores_all_classes.png`：最终各类别 sigmoid 分数。
- `raw_features.npz`：原始 dense 特征、聚合特征和最终分数，便于复核和重新绘图。
- `manifest.json`：权重 SHA256、输入路径、教师权重、张量形状、显示范围与类别顺序。

Dense 图采用逐 patch L2 归一化、空间中心化后的第一主成分（PC1），每个主干独立做 2–98 百分位显示缩放；另提供前三主成分 RGB 图。颜色表示特征变化，不是类别概率，三个模型的颜色不能直接作数值比较。CLIP 使用原始方向、进入分割头前的真实 dense 特征；分割推理保留配置中的四方向处理。RS-DINO 与 RemoteCLIP 是单独加载的冻结教师，不是目标分割检查点中的推理分支；RemoteCLIP 采用标准视觉前向的最终 patch tokens，经 ln_post 和投影，不包含 CLS，原生分辨率为 7×7。

空间和类别聚合在实现中是并行分支。两层各两次聚合均完整导出，主结果取最后一层第二次聚合、融合之前的输出。每个类别的热力图是隐藏通道 RMS，全部聚合图共享色标，不是语义概率。最终分割是未经修改的配置与检查点正常推理结果。所有放大均为显示插值，不提高原生特征分辨率。
''')
    print(json.dumps(manifest,indent=2), flush=True)


if __name__ == '__main__':
    main()

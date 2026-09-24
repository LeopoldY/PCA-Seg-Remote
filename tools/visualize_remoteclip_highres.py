#!/usr/bin/env python3
"""Recompute frozen RemoteCLIP patch features at a higher input resolution."""
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
from PIL import Image
from cat_seg.modeling.backbone.remote_clip import build_remote_clip_visual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=896)
    args = parser.parse_args()
    assert args.size > 224 and args.size % 32 == 0
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.source / 'manifest.json').read_text())
    weights = metadata['teacher_weights']['remoteclip']
    rgb = np.asarray(Image.open(metadata['image']).convert('RGB'))
    torch.manual_seed(0)
    torch.set_num_threads(4)
    visual = build_remote_clip_visual('ViT-B-32', weights).cuda().eval()
    visual.output_tokens = True
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().cuda()[None]
    mean = x.new_tensor([122.7709383, 116.7460125, 104.09373615])[None,:,None,None]
    std = x.new_tensor([68.5005327, 66.6321579, 70.323163])[None,:,None,None]
    x = F.interpolate((x-mean)/std, size=(args.size,args.size), mode='bilinear', align_corners=False)
    with torch.inference_mode():
        _, tokens = visual(x)
        dense = (visual.ln_post(tokens) @ visual.proj).float().cpu()
    grid = args.size // 32
    assert dense.shape == (1, grid*grid, 512) and torch.isfinite(dense).all()
    # Match the original RemoteCLIP visualization's PC1 extraction and scaling.
    centered = F.normalize(dense[0], dim=-1)
    centered -= centered.mean(0,keepdim=True)
    u,s,v = torch.linalg.svd(centered, full_matrices=False)
    components = u[:,:3] * s[:3]
    for j in range(3):
        if v[j, v[j].abs().argmax()] < 0:
            components[:,j] *= -1
    pca = components.numpy().reshape(grid,grid,3)
    low, high = np.percentile(pca,[2,98],axis=(0,1))
    normalized = np.clip((pca-low)/(high-low+1e-12),0,1)
    np.savez_compressed(args.output/'remoteclip_features.npz',dense=dense.numpy(),
                        pca=pca, normalized_pca=normalized)
    Image.fromarray(rgb).save(args.output/'input.png')
    def save_views(size, suffix):
        colors = cv2.resize(normalized, size, interpolation=cv2.INTER_LINEAR)
        heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(colors[:,:,0]*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
        background = cv2.resize(rgb,size,interpolation=cv2.INTER_LINEAR)
        overlay = np.uint8(background*.5+heat*.5)
        Image.fromarray(heat).save(args.output/f'remoteclip_heatmap_{suffix}.png')
        Image.fromarray(overlay).save(args.output/f'remoteclip_overlay_{suffix}.png')
        Image.fromarray(np.uint8(colors*255)).save(args.output/f'remoteclip_pca_rgb_{suffix}.png')
    save_views((rgb.shape[1],rgb.shape[0]), 'original_size')
    save_views((args.size,args.size),f'{args.size}px')
    native_heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(normalized[:,:,0]*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    Image.fromarray(native_heat).save(args.output/'remoteclip_native_grid.png')
    digest=hashlib.sha256()
    with open(weights,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    report={'image':metadata['image'],'original_size':[rgb.shape[1],rgb.shape[0]],
            'weights':weights,'teacher_sha256':digest.hexdigest(),
            'segmentation_checkpoint_context':metadata['checkpoint'],
            'input_resolution':[args.size,args.size], 'patch_size':32,
            'native_feature_grid':[grid,grid],'feature_shape':list(dense.shape),
            'method':'Standard visual forward; final patch tokens excluding CLS, ln_post then projection. Same frozen teacher and PC1 method as original visualization.',
            'positional_embedding':'Existing rescale_positional_embedding: bicubic 7x7 -> target grid, CLS unchanged.',
            'normalization':'Patch L2 normalization, spatial centering, PCA, sign fixed by largest loading; per-component P02-P98 clipping to [0,1].',
            'pc1_explained_variance':float(s[0]**2/(s**2).sum()),
            'note':'Features recomputed at higher input resolution; source image itself is 256x256. Finer feature sampling does not add original image detail.'}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    (args.output/'README.md').write_text(f'''# RemoteCLIP 高分辨率特征热力图

同一冻结 RemoteCLIP ViT-B/32 教师，将输入从 224×224 提升到 {args.size}×{args.size} 后重新前向计算，原生 dense 网格由 7×7 提升到 {grid}×{grid}。使用模型已有的双三次位置编码插值。热力图仍使用与旧版一致的 PC1 和 2–98 百分位归一化。

- `remoteclip_overlay_{args.size}px.png`：单张高分辨率叠加图。
- `remoteclip_heatmap_{args.size}px.png`：单张高分辨率纯热力图。
- `remoteclip_pca_rgb_{args.size}px.png`：前三主成分 RGB 图。
- `*_original_size.png`：原图尺寸版本，便于与其他模型结果对齐。
- `remoteclip_native_grid.png`：未经显示放大的原生特征网格图。
- `remoteclip_features.npz`：原始 dense 张量、PC 分量和归一化响应。
- `manifest.json`：权重校验值、真实网格尺寸和计算方法。

各图分别保存，无拼图。原始图像为 256×256；更大的模型输入增加特征采样密度，不增加原图细节。冻结教师使用训练配置指定的独立权重，不从分割检查点中加载教师参数。
''')
    files=list(args.output.glob('*.png'))
    assert len(files)==8
    for file in files:
        with Image.open(file) as im: im.verify()
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()

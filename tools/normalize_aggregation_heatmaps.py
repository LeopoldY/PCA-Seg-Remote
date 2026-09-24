#!/usr/bin/env python3
"""Render each captured aggregation class separately with robust normalization."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.source / 'manifest.json').read_text())
    rgb = np.asarray(Image.open(args.source / 'input.png').convert('RGB'))
    height, width = rgb.shape[:2]
    source = np.load(args.source / 'raw_features.npz')
    normalized_arrays = {}
    records = []
    for key in source.files:
        if not key.startswith('layer'):
            continue
        feature = source[key]
        assert feature.shape[:3] == (1, 128, 17)
        energy = np.sqrt(np.mean(np.square(feature[0]), axis=0))
        directory = args.output / key
        directory.mkdir(exist_ok=True)
        maps = []
        for index, name in enumerate(metadata['classes']):
            response = energy[index]
            low, high = np.percentile(response, [2, 98])
            if high - low <= 1e-8:
                normalized = np.zeros_like(response)
            else:
                normalized = np.clip((response - low) / (high - low), 0, 1)
            # A fixed gamma suppresses weak responses while preserving ordering.
            normalized = normalized ** 1.5
            maps.append(normalized)
            display = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_LINEAR)
            heat = cv2.applyColorMap(np.uint8(np.clip(display, 0, 1) * 255), cv2.COLORMAP_TURBO)
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
            overlay = np.uint8(np.clip(rgb * .35 + heat * .65, 0, 255))
            stem = f'{index:02d}_{name.replace(" ", "_")}'
            Image.fromarray(heat).save(directory / f'{stem}_heatmap.png')
            Image.fromarray(overlay).save(directory / f'{stem}_overlay.png')
            records.append({'stage': key, 'class_id': index, 'class': name,
                            'p02': float(low), 'p98': float(high),
                            'raw_min': float(response.min()), 'raw_max': float(response.max())})
        normalized_arrays[key] = np.stack(maps)
    np.savez_compressed(args.output / 'normalized_responses.npz', **normalized_arrays)
    metadata = {'source_checkpoint': metadata['checkpoint'],
                'checkpoint_sha256': metadata['checkpoint_sha256'],
                'source_features': str((args.source / 'raw_features.npz').resolve()),
                'normalization': 'Per stage and per class: clip((channel_RMS - P02)/(P98-P02), 0, 1)**1.5; constant maps become zero.',
                'overlay': '35% original RGB + 65% TURBO heatmap',
                'records': records}
    (args.output / 'normalization.json').write_text(json.dumps(metadata, indent=2))
    (args.output / 'README.md').write_text('''# 独立归一化的空间／类别聚合热力图

每张图只包含一个聚合阶段、一个类别。8 个阶段 × 17 类，每类分别保存纯热力图 `_heatmap.png` 和原图叠加 `_overlay.png`，共 272 张 PNG，不生成拼图。

主结果目录：`layer2_spatial_second`（最后一次空间聚合）、`layer2_class_second`（最后一次类别聚合）。其他目录保留之前的聚合阶段。

归一化：先计算各类别的隐藏通道 RMS，然后每张图独立以第 2、98 百分位作为下、上界裁剪并缩放到 [0,1]，再应用 gamma=1.5 压低弱响应。TURBO 色图中蓝／紫为低响应，黄／红为高响应。固定常量特征图显示为零。

这些是相对特征激活，不是类别概率。独立归一化突出各图内部差异，因此不同类别／阶段之间的颜色不能用于比较绝对激活大小；弱响应类别也可能显示局部高值。原始范围与归一化阈值记录在 `normalization.json`，原生 24×24 归一化响应保存在 `normalized_responses.npz`。原始特征来自指定检查点已完成的推理，没有重新训练或修改模型。
''')
    files = list(args.output.rglob('*.png'))
    assert len(files) == 272
    for file in files:
        with Image.open(file) as im:
            assert im.size == (width, height)
            im.verify()
    for value in normalized_arrays.values():
        assert np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
    print(f'Verified {len(files)} individual PNGs in {args.output}')


if __name__ == '__main__':
    main()

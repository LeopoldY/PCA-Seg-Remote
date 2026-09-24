#!/usr/bin/env python3
"""Export actual pre-convolution image/text cost volume, one image per class."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from cat_seg import add_cat_seg_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.source/'manifest.json').read_text())
    torch.manual_seed(0)
    torch.set_num_threads(4)
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(metadata['config'])
    cfg.MODEL.WEIGHTS = metadata['checkpoint']
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    model = predictor.model
    assert not model.sliding_window
    agg = model.sem_seg_head.predictor.transformer
    assert not agg.use_rs_dino
    classes = model.sem_seg_head.predictor.test_class_texts
    assert classes == metadata['classes'] and len(classes) == 17
    captured = []
    def capture(module, inputs):
        captured.append(inputs[0].detach().float().cpu().numpy().copy())
    hook = agg.conv1.register_forward_pre_hook(capture)
    bgr = cv2.imread(metadata['image'])
    assert bgr is not None
    rgb = bgr[:,:,::-1].copy()
    with torch.inference_mode():
        scores = predictor(bgr)['sem_seg'].float().cpu().numpy()
    hook.remove()
    assert len(captured) == 1
    raw = captured[0]  # (B*T, N*P, H, W), B=1; before learned conv1
    assert raw.shape[0] == 17 and np.isfinite(raw).all()
    directions = 4 if agg.use_clip_rotation else 1
    assert raw.shape[1] % directions == 0
    prompt_count = raw.shape[1] // directions
    raw = raw.reshape(17, directions, prompt_count, *raw.shape[-2:])
    # Preserve signed cosine similarity: an RMS would amplify negative matches.
    mean_cost = raw.mean(axis=(1,2))
    previous = np.load(args.source/'raw_features.npz')['segmentation_scores']
    max_diff = float(np.max(np.abs(scores-previous)))
    assert np.array_equal(scores.argmax(0), previous.argmax(0))
    assert np.allclose(scores,previous,atol=1e-5,rtol=1e-5), max_diff
    normalized_maps = []
    records = []
    for i, name in enumerate(classes):
        response = mean_cost[i]
        low,high = np.percentile(response,[2,98])
        normalized = (np.clip((response-low)/(high-low),0,1)**1.5
                      if high-low>1e-8 else np.zeros_like(response))
        normalized_maps.append(normalized)
        display = cv2.resize(normalized,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
        heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(display*255),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
        overlay = np.uint8(rgb*.35+heat*.65)
        stem=f'{i:02d}_{name.replace(" ","_")}'
        Image.fromarray(heat).save(args.output/f'{stem}_heatmap.png')
        Image.fromarray(overlay).save(args.output/f'{stem}_overlay.png')
        records.append({'class_id':i,'class':name,'p02':float(low),'p98':float(high),
                        'raw_min':float(response.min()),'raw_max':float(response.max())})
    normalized_maps=np.stack(normalized_maps)
    np.savez_compressed(args.output/'cost_volume.npz',raw_cost=raw,
                        mean_cost=mean_cost,normalized_cost=normalized_maps)
    report={'checkpoint':metadata['checkpoint'],'checkpoint_sha256_from_source':metadata['checkpoint_sha256'],
            'image':metadata['image'],'classes':classes,'raw_shape':list(raw.shape),
            'raw_axes':['class','aligned_rotation','prompt','height','width'],
            'capture':'Actual input of aggregator.conv1 in normal inference; after attribute_adapter and image/text cosine correlation, before cost embedding and spatial/class aggregation.',
            'class_reduction':'Arithmetic mean of signed cosine similarities over aligned rotation and prompt dimensions; no absolute value or RMS.',
            'normalization':'Per class: clip((mean_cost-P02)/(P98-P02),0,1)**1.5; constant maps zero.',
            'prediction_check':{'same_argmax_as_original':True,'max_score_abs_diff':max_diff},
            'records':records}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
    (args.output/'README.md').write_text(f'''# Cost volume 逐类别热力图

17 个类别，每类单独保存纯热力图 `*_heatmap.png` 与叠加图 `*_overlay.png`，共 34 张，无拼图。

使用指定 model_0054999.pth 正常推理，捕获聚合模块 `conv1` 的真实输入，即属性适配后的图文余弦相似度 Cost volume，位于代价嵌入、空间聚合和类别聚合之前。四方向特征已旋转回原图坐标。原始张量按 `[类别, 方向, 提示, H, W]` 保存，形状 `{list(raw.shape)}`。

每类在方向和提示维度取有符号余弦相似度的算术平均；不取绝对值或 RMS，避免把负相似度显示为高匹配。之后每类独立进行 2–98 百分位裁剪归一化和 gamma=1.5 弱响应抑制，蓝／紫为相对低匹配，黄／红为相对高匹配。各类别独立归一化，颜色不代表类别概率，不能跨类别比较绝对相似度。

`cost_volume.npz` 保留未经降维的原始 Cost volume、逐类均值及归一化热力图。`manifest.json` 记录提取位置、数值范围及推理一致性校验。此次推理与最初分割结果类别 ID 完全一致，最大分数差为 {max_diff:.8g}。
''')
    files=list(args.output.glob('*.png'))
    assert len(files)==34
    for file in files:
        with Image.open(file) as image:
            assert image.size == (rgb.shape[1],rgb.shape[0])
            image.verify()
    assert np.isfinite(normalized_maps).all()
    print(json.dumps({'verified_pngs':len(files),'raw_shape':list(raw.shape),
                      'prediction_check':report['prediction_check']},indent=2),flush=True)


if __name__=='__main__':
    main()

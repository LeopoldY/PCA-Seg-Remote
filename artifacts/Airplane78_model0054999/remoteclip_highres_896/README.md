# RemoteCLIP 高分辨率特征热力图

同一冻结 RemoteCLIP ViT-B/32 教师，将输入从 224×224 提升到 896×896 后重新前向计算，原生 dense 网格由 7×7 提升到 28×28。使用模型已有的双三次位置编码插值。热力图仍使用与旧版一致的 PC1 和 2–98 百分位归一化。

- `remoteclip_overlay_896px.png`：单张高分辨率叠加图。
- `remoteclip_heatmap_896px.png`：单张高分辨率纯热力图。
- `remoteclip_pca_rgb_896px.png`：前三主成分 RGB 图。
- `*_original_size.png`：原图尺寸版本，便于与其他模型结果对齐。
- `remoteclip_native_grid.png`：未经显示放大的原生特征网格图。
- `remoteclip_features.npz`：原始 dense 张量、PC 分量和归一化响应。
- `manifest.json`：权重校验值、真实网格尺寸和计算方法。

各图分别保存，无拼图。原始图像为 256×256；更大的模型输入增加特征采样密度，不增加原图细节。冻结教师使用训练配置指定的独立权重，不从分割检查点中加载教师参数。

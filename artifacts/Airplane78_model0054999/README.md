# Airplane78 特征与分割可视化

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

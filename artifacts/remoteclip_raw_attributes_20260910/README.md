# 原始属性文本向量可视化

编码器为 RemoteCLIP ViT-B-32 和 ViT-L-14。DLRSD 为 17 类、340 条文本；iSAID 为 15 类、300 条文本，每类 20 条。

直接读取 attributes_text/{dataset}_train_descriptors.json 中的完整原句，逐条调用 RemoteCLIP encode_text 并 L2 归一化，不进行额外大小写转换，不执行聚类，也不读取属性中心库。每个散点对应一句原始描述，颜色表示来源类别。同一数据集的两种模型使用相同类别配色。

raw_attributes_tsne.png/pdf：四组向量独立进行 t-SNE，perplexity=30、init=pca、learning_rate=auto、random_state=42，使用归一化向量的欧氏距离。局部邻域用于探索；图间坐标和簇间距离不可直接比较。

raw_attributes_pca.png/pdf：四组向量独立进行二维 PCA，坐标轴注明解释方差，用于补充线性投影视角。

每组原始向量保存在 *_raw.pth 中，包含 embeddings [N,D]、labels、classes 和未经修改的 texts。projections.json 包含逐点文本、类别、二维坐标、PCA 解释方差及源文件 SHA256。

编码与降维（服务器）：/opt/miniconda3/envs/yc_d2/bin/python tools/visualize_remoteclip_attributes.py

绘图（具备 matplotlib 和 numpy 的环境）：python tools/plot_remoteclip_attributes.py

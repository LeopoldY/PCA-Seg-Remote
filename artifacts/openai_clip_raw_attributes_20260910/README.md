# OpenAI CLIP 原始属性文本可视化

使用当前项目学生初始化所对应的 OpenAI CLIP ViT-B/16、ViT-L/14@336px 权重。DLRSD：17 类、340 句；iSAID：15 类、300 句；每类 20 句。

逐条编码原始 JSON 中的完整描述，保留原文，L2 归一化，不使用聚类中心。原始向量、类别、文本分别保存在 *_raw.pth；二维坐标、文本、类别、来源哈希、权重路径见 projections.json。

与 RemoteCLIP 图沿用相同类别颜色和参数：t-SNE perplexity=30、init=pca、learning_rate=auto、random_state=42；PCA 独立拟合两维，并标注解释方差。不同面板坐标与距离不可直接比较。OpenAI B/16 与此前 RemoteCLIP B/32 的图像架构不同，因此这不是完全相同骨干的受控比较。

编码脚本：tools/visualize_openai_clip_attributes.py（服务器 CPU float32）。绘图脚本：tools/plot_openai_clip_attributes.py（matplotlib）。

图像输出：raw_attributes_tsne.png/pdf、raw_attributes_pca.png/pdf。

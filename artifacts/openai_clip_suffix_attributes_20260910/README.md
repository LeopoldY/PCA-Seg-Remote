# 去掉类别前缀后的属性文本可视化

从每条描述开头删除 `an aerial image of XXX.` 及紧随的空白，保留后半句原文（包括 `it has`、`its` 等）。不额外删除后半句中出现的类别或物体名。原始属性 JSON 和 20/64 中心属性库保持不变。

例如 `an aerial image of airplane. it has a long narrow fuselage with two lateral wings.` 转为 `it has a long narrow fuselage with two lateral wings.`。

每个点为一条后半句的 L2 归一化文本向量，颜色仍对应原始语义类别。DLRSD 为 340 点、17 类；iSAID 为 300 点、15 类，不做聚类或去重。去掉前缀后不同类别可出现相同文本，保留各自标签。

t-SNE 使用 perplexity=30、init=pca、learning_rate=auto、random_state=42；PCA 为二维线性投影。参数和配色与完整句子版本一致，各面板独立拟合，跨图坐标和距离不可直接比较。

*_suffix.pth 保存 embeddings、labels、classes、texts（后半句）、original_texts（原句）；projections.json 保存对应文本和二维坐标。另存两套 *_attribute_suffixes.json 供查看。

结果为 suffix_attributes_tsne.png/pdf 和 suffix_attributes_pca.png/pdf。

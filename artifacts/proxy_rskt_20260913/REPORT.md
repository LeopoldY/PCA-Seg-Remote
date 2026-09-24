# ProxyDet 思路迁移：训练属性代理库与 RemoteCLIP 比较

实验日期：2026-09-13。编码与投影运行于 **inspur，物理 GPU 1**；输出已同步回当前工作区。

## 方法核对

[ProxyDet 论文](https://arxiv.org/abs/2312.07266) 与[官方 mixup 实现](https://github.com/clovaai/ProxyDet/blob/033c51ccece41342d89f2a6ac915f831f018f527/proxydet/modeling/roi_heads/proxydet_roi_heads.py#L183)表明：代理类别是在嵌入空间中形成的虚拟类别，不是通过语言模型生成新类别名称。官方实现从当前正类建立视觉原型，取对应的文本向量，随机打乱配对；用同一个 λ 混合两种模态，再分别 L2 归一化，并计算代理对齐损失。默认 λ 服从 Beta(1,1)。核对的提交为 `033c51ccece41342d89f2a6ac915f831f018f527`。

本实验是**属性文本空间的离线迁移**，未实现或运行检测器的视觉原型、代理损失或分割训练。对每个训练类别 c 的第 k 条属性：

```
e[c,k] = normalize(mean_t(normalize(RemoteCLIP(prompt[t,c,k]))))
p[c]   = normalize(mean_k(e[c,k]))
z[a,b] = normalize(lambda * p[a] + (1-lambda) * p[b])
expanded = concatenate(all_original_e, all_proxy_z)
```

DLRSD、iSAID 分别建库。枚举各自所有不重复的异类配对，每对独立采样 4 个 Beta(1,1) 系数，Python 随机种子 42；这是本实验的静态采样设定，不是官方逐批随机配对调度。没有根据目标类别挑选父类、系数或代理。代理 ID 和来源是生成配方，不应将两个父类名称拼接后再次作为代理文本编码。

## 优先创建的扩充库

| 训练来源 | 原类别数 | 完整原属性数 | 代理数 | 扩充条目总数 |
|---|---:|---:|---:|---:|
| DLRSD | 17 | 340 | 544 | 884 |
| iSAID | 15 | 300 | 420 | 720 |

文本及配方位于 `../../attributes_text/proxy_rskt/DLRSD_expanded.json` 和 `../../attributes_text/proxy_rskt/iSAID_expanded.json`。保留每条原句原文、顺序、父类、λ、提示词、源文件 SHA256。代理本体在编码后保存为向量；JSON 可在无模型环境下先创建。

每套训练库均生成 B/32（512 维）、L/14（768 维）两个版本：

- `*_expanded.pth`：完整元数据、原属性向量、类别属性原型、代理向量、拼接后的 `embeddings`。前 N 行为重新按 RS 模板编码的原属性，后 M 行为代理。
- `*_embedding_bank.pth`：当前 ExCEL loader 可读取的 `[bank, class_flags]`；bank 为 `[D,N+M]`。未做聚类，完整保留原属性向量。代理的 flags 标记两个训练父类，具体权重在完整文件中。
- 模型训练配置若指定了 20/64 个属性槽位，需要另行改为 884/720 或设计聚类版本；本次没有更改训练配置或启动训练。

这里“保留原内容”指原文完整保留且其本次编码结果作为扩充库前缀保留，不是复用旧的无模板编码结果或原 20/64 聚类中心。

## RSKT-SEG 提示词

使用 inspur 上 `RSKT_Seg/third_party/imagenet_templates.py` 的 **RS_TEMPLATES 全部 8 条**；已逐条断言一致。原定义中的 `satallite` 拼写也原样保留。该文件另有 14 条 RS_ALL_TEMPLATES，本次未选该集合。

原属性形如 `an aerial image of airplane. it has ...`，编码时以每条 RS 模板包裹其原类别名称，再接原属性后缀 `it has ...`，避免将一整句嵌入 `{}` 中产生嵌套句式。JSON 中仍保留完整原句。目标类别只将其原始类别名填入相同模板。两者均先逐模板归一化，再平均并归一化。

## 跨数据集比较结果

分别与另外 7 个数据集比较：DLRSD / iSAID、LoveDA、Potsdam、Vaihingen、UDD5、VDD、uavid。剔除 no-data、background、clutter、other 等非具体语义类别。比较仍包含共享类别和近义标签；CSV 的 `exact_train_name_overlap` 仅作忽略大小写的同名检查，**不是完整未知类别判定**。

下表对“数据集—类别”行求均值；重复类别在不同数据集中会重复计数，因此不是去重后未知类别覆盖率。增益为扩充库最高余弦减去原属性库最高余弦。

| 训练库 | 模型 | 原属性最佳余弦均值 | 类别原型最佳余弦均值 | 仅代理最佳余弦均值 | 扩充增益均值 | 增益 > 1e-6 的行数 |
|---|---|---:|---:|---:|---:|---:|
| DLRSD | ViT-B-32 | 0.8465 | 0.8536 | 0.8750 | 0.0292 | 45/48 |
| iSAID | ViT-B-32 | 0.8358 | 0.8303 | 0.8608 | 0.0251 | 48/50 |
| DLRSD | ViT-L-14 | 0.8933 | 0.8761 | 0.8911 | 0.0048 | 22/48 |
| iSAID | ViT-L-14 | 0.8743 | 0.8720 | 0.8972 | 0.0235 | 48/50 |

B/32 的两套库均观察到较多行的几何覆盖增益；DLRSD 的 L/14 增益较小，并且“仅代理”均值低于原属性库。因此应保留原属性，不能直接以代理替换。父类组合可能没有直接的语言学含义，例如最接近 Agriculture 的代理也可能来自 ship 与 vehicle；它只表示文本嵌入中的方向插值，不能据此认定构造出真实的农业概念。

扩充库包含原库，最大相似度必然不下降，且向量数量增加会影响最近邻统计。这些数据是特征空间诊断，不是未知类别识别率、mIoU 或泛化提升证明。正式验证需要后续分割实验及等规模采样对照。

## 图表和复现

每套训练库、每种模型均提供 `*_pca.png/pdf`、`*_tsne.png/pdf`、`*_cosine.png/pdf` 和 `*_cosine_comparison.csv`，总计 12 组图。橙色为代理，灰色为原属性，蓝色方块为训练属性原型，红星为比较类别。

PCA 只在原属性、代理、训练属性原型上拟合，再投影目标类别；同一图中 7 个面板共享坐标。t-SNE 使用一次联合拟合，包含全部比较类别名称的去重向量，7 个面板也共享坐标。不同训练库或不同模型图间的坐标不可直接比较。t-SNE 只用于局部邻域展示；定量结论使用完整 512/768 维空间余弦。

脚本为 `../../tools/build_proxy_attribute_bank.py`、`../../tools/plot_proxy_attribute_bank.py`。远端输入快照位于本目录 `run_input/`，源描述与本地此次输入一致。

在 inspur 执行：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 /opt/miniconda3/envs/yc_d2/bin/python /mnt/data6/yc/open-vocab/PCA-Seg-Remote/artifacts/proxy_rskt_20260913/run_input/tools/build_proxy_attribute_bank.py --encode --checkpoint-dir /mnt/data6/yc/open-vocab/PCA-Seg-Remote/pretrained --output /mnt/data6/yc/open-vocab/PCA-Seg-Remote/artifacts/proxy_rskt_20260913
/opt/miniconda3/envs/yc_declip/bin/python /mnt/data6/yc/open-vocab/PCA-Seg-Remote/artifacts/proxy_rskt_20260913/run_input/tools/plot_proxy_attribute_bank.py --output /mnt/data6/yc/open-vocab/PCA-Seg-Remote/artifacts/proxy_rskt_20260913
```

编码使用 yc_d2，绘图使用已有 yc_declip，避免改动服务器训练环境依赖。已独立逐条重算所有代理、验证原文、原向量前缀、单位范数、有限值和 ExCEL bank 一致性。详见 `validation.json`、`template_validation.json`、`encoding.log`。权重 SHA256 存于每个完整 PTH 的元数据中。

# RemoteCLIP → CLIP 蒸馏诊断与实验建议

日期：2026-09-13。依据：当前本地实现、已归档的完整训练指标与配置、论文原始来源。未启动训练，未修改训练代码或现有配置；本地 pretrained 目录无可用于本次特征实测的权重。下文将实现事实、历史观测、待验证假设分开。

## 判断

“两个模型的特征差异过大，直接蒸馏破坏 CLIP 表达”是合理机制，但目前不能确认为 iSAID 表现差的主因。更精确的假设是：不同视觉—文本空间之间的坐标约束、区域监督污染、训练目标权重与数据分布共同造成负迁移。低跨模型余弦本身不等于信息差，也不等于能力损坏；必须检查同模型文本匹配能力、跨数据集表现和有无 RemoteCLIP 的受控对照。

## 实现审计

1. `cat_seg/cat_seg_model.py:562` 与 `cat_seg/modeling/backbone/remote_clip.py:30`：旧路径以区域裁剪的教师 CLS 向量监督整图学生 ROI 特征；新 SAME_GRID 路径让教师 768/B32 与学生 384/B16 都输出 24×24，再对相同框池化。两者均直接计算 `1-cos(student, teacher)`，没有可学习的跨模型映射，也没有冻结 OpenAI CLIP 的保留目标。新路径解决网格与视野对应，不保证语义坐标一致、稠密教师质量或有效空间带宽一致。上采样不能恢复原图没有的小目标信息，7×7 预训练位置编码插值到 24×24 的效果也尚未验证。
2. `cat_seg/cat_seg_model.py:511`：每个类别只取一个覆盖所有像素的包围框；按该类别像素数排序、最多取 8 类、框边长至少 8。离散小车/船等可合并成横跨整图的大框；ROI 会包含其他类和未标注像素。边长检查不能保证前景占比，也不能保证学生网格上有足够有效像素。该机制可能使 iSAID 的内容监督偏向场景背景，实际污染程度尚需统计。
3. `cat_seg/modeling/attribute_fusion/attribute_text_adapter.py:56`：直接用 OpenAI 类别文本与属性库点积，再将属性加权向量与原文本相加。当前 ViT-B 配置指向 RemoteCLIP B32 属性库，没有空间映射。512 维相等只保证运算可执行；这一混用即便关闭 RemoteCLIP 图像蒸馏仍然存在。应单独消融。优先将同一批描述用 OpenAI 文本塔重新编码，控制描述和聚类归属；或由 RemoteCLIP 决定属性权重，再用配对的 OpenAI 属性向量进行聚合。
4. `cat_seg/cat_seg_model.py:683` 附近：mask 只限制 one-hot 写入，BCE 仍对全部像素求均值；ignore 像素的目标是全零。也就是说，ignore 在训练中成为所有类的负样本。若 255 明确代表已知背景，这可能是有意的负监督；若包含未知对象、无效区域或 padding，则不符合 ignore 语义。需分开这些含义，并记录比例。有效像素 BCE 可用 `BCE(outputs[mask], onehot)` 实现，同时处理全 ignore 样本；不能不经消融把当前背景负监督全部删掉。
5. 教师冻结、eval、no_grad 路径存在，未见教师误更新。QV 部分微调仍会改变学生特征，不自动保留语言对齐。分割使用四方向特征，content 监督原始方向；共享参数会将梯度传播到所有方向的编码行为。

## 历史证据及适用范围

`artifacts/rskt_clipl_results_20260909/{DLRSD,iSAID}/metrics.json` 各有 3000 条训练记录。以下均为窗口内日志记录均值；content 除以该运行保存配置中的 5e-5，恢复未加权值。日志自身可能含平滑，非逐 batch 原始值。

| 数据集 | 窗口（iteration） | 原始 content | 加权分割 BCE |
|---|---:|---:|---:|
| iSAID | 0–999 | 0.311592 | 0.031591 |
| iSAID | 14000–14999 | 0.314037 | 0.006561 |
| iSAID | 29000–29999 | 0.297594 | 0.004876 |
| iSAID | 59000–59999 | 0.287161 | 0.003964 |
| DLRSD | 0–999 | 0.369072 | 0.072806 |
| DLRSD | 59000–59999 | 0.376296 | 0.006578 |

iSAID content 下降约 7.84%，对应平均余弦约 0.688→0.713。DLRSD content 未比初期更低，但域内验证 mIoU 持续上升。因此“content 降得慢→模型更差”并不是这些日志支持的普遍关系。

iSAID 验证最佳 85.9191（30k），最终 85.3530（60k）；DLRSD 最佳 92.3485。六个共同外部数据集均值分别为 26.0733、39.2396。这支持存在泛化差异，但训练来源、类别覆盖不同，不能作为 RemoteCLIP 的因果消融。`*_all_sem_seg` 与训练验证 split 不同；尤其 iSAID all 的 95.5291 不应当作独立泛化指标。

上述 ViT-L 运行保存配置使用 OpenAI 属性库、RemoteCLIP L14 crop 教师和 AFFResidualMoE。当前 ViT-B RS 配置使用 RemoteCLIP 属性库、SAME_GRID 与 DualFeatureMoE。两者不可混为同一实验。ViT-B AFF 基础配置当前 content/context=1/0.1，而 DualFeatureMoE 子配置覆盖为 1e-4/1e-5；必须读取运行保存配置，不能只看父配置或文件名。随机输入 SAME_GRID loss≈0.978 是流程测试，不能用于判断真实图像上的语义差异。

## 为什么直接余弦可能产生负迁移

设学生与 OpenAI 文本原型的分数为 `z_s = f_s T_openai^T`。直接迫使 `f_s → f_remote`，并不保证 `f_remote T_openai^T` 仍有正确语义。RemoteCLIP 应首先与其自身文本塔配对。极端地，若两个模型仅相差正交变换 R，图文两塔同时变换后检索能力不变，但跨空间直接余弦可以很低。L2 归一化不能消除这种坐标差异。

RemoteCLIP 官方验证了遥感分类、检索等能力，不保证其任意最后层 V 分支就是可靠像素教师。因此应先验证教师在目标区域上的识别/排序是否优于 OpenAI，而非以模型名称决定监督可信度。[RemoteCLIP 官方论文](https://arxiv.org/abs/2306.11029)

## 首选方案：语义分布蒸馏 + 原 CLIP 保留 + 可靠区域

这是基于文献与代码问题提出的组合方案，尚未在本项目验证。

对同一语义词表 C、同一有效区域 r，分别计算：

`z_s(r,c) = a_s cos(f_s(r), T_openai(c))`

`z_R(r,c) = a_R cos(f_R(r), T_remote(c))`

`L_RS = sum_r w_r τ² KL(softmax(z_R/τ) || softmax(z_s/τ)) / (sum_r w_r + eps)`

学生保留 OpenAI 文本空间，教师使用 RemoteCLIP 自己的文本空间；跨模型只对齐词义一一对应的概率分布，不要求特征坐标相等。初版可固定同一 logit scale 控制变量，随后在验证集校准 scale/温度，避免只蒸馏教师置信度差异。词表使用训练允许的类别、遥感描述与通用概念；若仅有 15 类则保留覆盖很有限。遵守 benchmark 对未见类名称和属性的使用约定，不使用测试图像或标签调参。

引入冻结的训练初始化学生副本 f0（OpenAI 学生时即原 CLIP；EVA/DeCLIP 初始化时应先保留实际起点，不能假定它等同 OpenAI），在通用与遥感描述上计算：

`L_keep = τ0² KL(p0 || ps)`

`L_total = L_seg + λ_RS L_RS + λ_keep L_keep + λ_D L_RS-DINO`

也可在同一原 CLIP 空间中补充区域/全局 cosine 保留，但不能把所有位置严格锁死而阻碍域适应。通用图像回放可扩大保留范围；只在 iSAID 图像上加保留项，不保证自然图像域能力完整保留。保留教师仅训练使用，推理不新增图像教师。

区域构造先改成连通域或实例级，再使用前景 mask 的面积占比加权池化；同类分散实例不要合并为一个框。对小于网格分辨率的实例使用独立的局部高分辨率视图，或跳过不可靠监督。类别均衡采样，记录面积、前景纯度、教师预测正确率与置信度。仅凭低熵筛选会保留“自信但错误”的教师；训练有 GT 时可联合正确性、多视图稳定性决定 w_r，并记录每类保留率。教师输出和 w_r 都 detach。

先冻结 CLIP 主干训练头/轻量残差适配器，再在确有收益时小学习率解冻少量 QV。残差适配器若负责承载遥感知识，应保留在推理路径；只有训练期 projector loss 下降而主干/推理不变，不能称为有效蒸馏。

PromptKD 通过文本原型与 KL 进行 CLIP 蒸馏，且采用视觉 prompt 与 projector；原方法共享教师文本向量，不能直接说它证明了上述“各自文本塔”方案。这里借鉴其输出分布蒸馏思想并为保留 OpenAI 空间作调整。[PromptKD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptKD_Unsupervised_Prompt_Distillation_for_Vision-Language_Models_CVPR_2024_paper.html)

Lipsum-FT 研究 CLIP 微调的特征失真与语言引导保留；可借鉴其用更广文本约束图文行为的思路，不能把本方案的 KL 当作该文原公式。[Lipsum-FT, ICLR 2024](https://arxiv.org/abs/2404.00860)

## 备选方案与使用条件

| 方法 | 项目落地 | 限制 |
|---|---|---|
| 关系蒸馏 | 对可靠区域归一化特征的两两 Gram/相似度矩阵做匹配；或蒸馏区域间距离/角度 | 不要求坐标对齐，但不能独立锚定词义；与 RS-DINO 空间关系可能冗余，初版不同时叠加 |
| 受限 projector | `1-cos(P(f_s), f_R)`；先冻结主干拟合线性映射，再弱解冻并保留原 CLIP | projector 可独自吸收全部差异；需单独检查主干增益，避免大 MLP 把 loss 降低误认为知识迁移 |
| 梯度保护 | 在共享 QV 上测量 RS、分割、keep、DINO 梯度夹角，对持续冲突的 RS 梯度作投影/降权 | 属局部一阶保护，非长期能力保证；严格初始化处 keep 梯度可能为零，需显式处理 |
| 权重插值 | 同结构 CLIP 的初始化与微调视觉权重做插值，并验证聚合头兼容性 | 只能在匹配的参数间操作，不能插值 OpenAI B16 与 RemoteCLIP B32；对新头无零样本对应参数 |

关系蒸馏依据：[RKD, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.pdf)、[Similarity-Preserving KD, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.html)。本文所提区域 Gram 是关系保持的具体改造，不是 RKD 原公式。

梯度保护依据：[ProGrad, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Prompt-aligned_Gradient_for_Prompt_Tuning_ICCV_2023_paper.pdf)，其原应用是 prompt tuning，迁移到 QV 和多教师需重新验证。

权重插值依据：[WiSE-FT](https://arxiv.org/abs/2109.01903)。

保留 content/context 分工仍有合理性，但 DeCLIP 对裁剪内容与结构上下文的监督不能推出“替换为任意异空间视觉教师仍可直接 cosine”的结论。[DeCLIP, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DeCLIP_Decoupled_Learning_for_Open-Vocabulary_Dense_Perception_CVPR_2025_paper.pdf)

## 最小诊断与因果实验

先离线固定一批训练图像和独立验证图像，不更新参数：

- 在同一视图/区域、对应 dense 提取路径下测量真实特征 cosine、线性 CKA、区域检索和类别排序；拟合只见训练区域的线性/正交映射并在验证区域测试。映射显著改善对齐而同模型语义准确率原本良好，支持坐标差异；不要用训练拟合误差作结论。
- 分别比较 OpenAI视觉×OpenAI文本、Remote视觉×Remote文本、Remote视觉×OpenAI文本；另测 OpenAI文本与 Remote 属性混用的影响。若 Remote 自身识别在小目标上就更差，该区域不应强蒸馏。
- 在共享 QV 上分别 `autograd.grad`，记录未加权与实际加权梯度范数、两两余弦及各层分布；冻结参数排除，unused 梯度正确处理。不能仅按 loss 数值与 organ 对齐来定权重。记录最终梯度裁剪是否经常触发。
- 统计各类 ignore 比例、实例面积、bbox 前景纯度、每类采样次数、有效网格覆盖；核查学生与教师 resize/padding/旋转坐标和训练标签映射。

受控训练先固定当前 backbone、初始化、数据划分、采样、提示词、属性、MoE、batch、LR 与预算：

| 组 | 教师/目标 | 回答问题 |
|---|---|---|
| A | 无蒸馏 | 基线 |
| B | 仅 RS-DINO | context 单独影响 |
| C | 仅 RemoteCLIP 当前 cosine | content 单独影响 |
| D | 当前双教师 | 教师交互是否造成损害 |
| E | D + 可靠 mask 区域 | 监督污染是否主因 |
| F | E 中 cosine 换各自文本塔的语义 KL | 避免坐标约束是否有效 |
| G | F + 原 CLIP 保留 | 是否缓解遗忘 |

属性空间、ignore 处理分别作为 D 或最优组的一因素对照，不同时替换所有组件。先短程筛选，再将有希望的配置跑足相同预算、至少三个随机种子。短程筛选不能保证最终排序。

评价同时记录 iSAID 留出验证、每类与小目标指标、预先指定的外部验证泛化、固定原 CLIP 图文探针；最终测试集只用于最终报告。严禁拿训练来源 all 集合高分证明泛化。若冻结原 CLIP 探针下降、关闭 Remote 恢复、加入 keep 再恢复且区域/属性等控制一致，才有更强证据认定 Remote 导致表达能力损失。

损失权重先按共享参数梯度而非标量大小校准；可将 RS 梯度相对分割梯度 0.05/0.1/0.2 作为待试起点，使用上限和稳定滑动统计，分割梯度近零时不机械放大权重。这是搜索建议，不是已验证最佳超参。逐步启用蒸馏，记录温度、筛选比例与原始 loss；不要以 content 越接近零越好作为优化或选模原则。

优先级：核实当前运行与权重 → 属性同空间/ignore 审计 → 可靠区域与梯度诊断 → 语义 KL + CLIP 保留 → 按证据增加关系蒸馏或梯度投影。无需第一版堆叠全部方法。

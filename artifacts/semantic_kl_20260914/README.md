# 同语义概念蒸馏实现与验证

日期：2026-09-14。部署目录：inspur `/mnt/data6/yc/open-vocab/PCA-Seg-Remote`。

原 iSAID SAME_GRID 正式训练 PID 2605817 及其全部子进程已停止，原输出和检查点保留。本次仅运行校准和小规模验证，未启动新的 60,000 步正式训练。修改前源码备份在服务器本目录 `backup/`。

## 实现

- 新配置项 `REMOTE_CLIP_DISTILL.OBJECTIVE=semantic_kl`；默认仍是 `cosine`，旧实验配置保持兼容。
- 同一增强后完整图像：OpenAI ViT-B/16 输入 384，RemoteCLIP B/32 输入 768，均生成 24×24 网格。相同语义区域框、相同 ROIAlign、逐位置及池化后归一化。
- 使用训练类 JSON 中相同且有序的概念：iSAID 15 类、DLRSD 17 类。师生分别使用 RemoteCLIP、OpenAI 的对应文本塔；使用同一组 RS 8 模板，模板/别名归一化聚合策略一致。不是混用两种模型的特征坐标，也不使用测试类别生成教师目标。
- 学生文本锚点来自未做属性融合的 OpenAI 类别嵌入；教师文本锚点首次训练前向时编码一次，随后释放教师文本塔。视觉教师冻结、不注册到学生模块树；两个文本锚点为不持久化 buffer，不进入 checkpoint。
- 目标：`T² * KL(softmax(10 * cos(f_remote,t_remote)/T) || softmax(10 * cos(f_student,t_openai)/T))`，T=2。按概念求和、按区域求均值。teacher 和文本锚点 detach，梯度只进入学生视觉特征。
- 分割属性库切换到两套 `*_train_desc_openai_b16_cluster_64_embedding_bank.pth`，已从服务器复制到本地，保持 TOP_K=0.9。类别分布蒸馏不使用属性融合后的文本。
- 新损失名 `loss_remote_clip_semantic`；原始 KL 通过 `remote_clip_semantic_kl_raw` 单独记录，不再次计入 total_loss。最终代码对该诊断值执行跨 rank 均值，使其与加权 loss 口径一致。
- 本次不额外加入原 CLIP 图像保留教师，也未改动区域构造或 ignore BCE 语义；这些属于此前报告的独立消融项。本次实现语义蒸馏、OpenAI 属性库和损失校准。

## 权重与实测

先分别用固定 seed=42 的真实训练数据完成 4 batch × 4 图的零更新校准；第一 batch 验证语义 KL 独立反向传播。校准日志最初使用 semantic=1e-4、context=1.5e-5；随后根据原始 loss 选择以下固定权重，不使用动态归一化强行保持 loss。

| 数据集 | 原始 KL 校准均值 | 最终 KL 系数 | 最终 context 系数 | 双卡 smoke 加权 KL | 双卡 smoke 加权 context |
|---|---:|---:|---:|---:|---:|
| iSAID | 0.0248679 | 0.004 | 0.000016 | 0.00009057 | 0.00011241 |
| DLRSD | 0.0157691 | 0.006 | 0.000016 | 0.00010749 | 0.00009717 |

两项蒸馏损失初始化附近处于 1e-4 数量级。训练中允许它们自然变化；不是将所有分割/正交损失或总损失缩放到 1e-4。

表中 iSAID 来自 `smoke_iSAID_final/metrics.json`，DLRSD 来自 `smoke_DLRSD/metrics.json`，均为 2 步训练的日志平滑值。旧 smoke 中原始 KL 诊断是 rank0 局部值；最终实现已更正跨 rank 归约，iSAID 最终复测验证 weighted/raw≈0.004。DLRSD 原 smoke 的加权 loss 本身已是正确的双卡均值。

## 验证

- 数学测试：独立旋转两侧坐标后语义判断相同则 KL≈0；允许师生不同嵌入维度；显式验证 KL 方向、T²、归约方式；学生梯度非零且有限，教师和文本无梯度；非法温度被拒绝。
- 兼容测试：原同网格 cosine 在相同特征/区域下近零，不匹配网格报错。
- 两套真实数据校准均退出码 0，optimizer_steps=0，教师冻结且不进入学生参数和 state_dict。
- iSAID、DLRSD 均通过 GPU 2、3 的 2 步 DDP 训练、保存 model_final.pth 和 4 张验证图推理。校准后的实际损失见上表。iSAID 在诊断日志归约修改后再次通过完整 smoke。
- 两个正式训练入口已 dry-run 验证配置、OpenAI 属性库路径和 CUDA_VISIBLE_DEVICES=2,3。
- 新进程加载最终 iSAID smoke checkpoint，将两个教师权重路径设为不存在，双卡 eval-only 仍成功退出（0）；证明推理不加载教师，见 `inference_without_teachers.log`。检查后 GPU 2、3 各约 409 MiB，无本次训练或验证占用。
- smoke 验证集缺失类别会出现 NaN 类指标，不是训练 loss NaN；两步 smoke 精度不用于声称正式精度提升。退出时存在原训练框架的 NCCL 未显式 destroy 警告，训练进程均正常退出。
- 本地只保存日志、配置与校准数据；smoke checkpoint 留在服务器，不下载大权重。

## 训练入口

服务器执行，先 iSAID 后 DLRSD（串行复用 GPU 2、3）：

```bash
cd /mnt/data6/yc/open-vocab/PCA-Seg-Remote
bash scripts/train_semantic_kl_isaid_first_gpu23.sh
```

单独执行：

```bash
bash /mnt/data6/yc/open-vocab/PCA-Seg-Remote/scripts/train_isaid_semantic_kl_gpu23.sh
bash /mnt/data6/yc/open-vocab/PCA-Seg-Remote/scripts/train_dlrsd_semantic_kl_gpu23.sh
```

不要同时执行两个单独入口争用同一组 GPU。默认全局 batch=8（每 GPU 4）、学习率 2e-4、60,000 步、四方向、DualFeatureMoE、RS 8 模板、OpenAI 64 聚类属性。每次生成独立带时间戳的输出目录，默认从 OpenAI 初始化训练，不续接旧 cosine 实验。显式恢复已有同方案运行时，通用单数据集入口支持 RESUME=1 与 OUTPUT_DIR；顺序入口默认启动新实验。

独立校准：`PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python scripts/check_semantic_kl.py --config configs/clip_vitb_384_isaid_attr64_semantic_kl.yaml --output calibration.json`，需使用训练环境及其数据集环境变量。

双卡 smoke：`bash scripts/smoke_semantic_kl_gpu23.sh iSAID` 或 `DLRSD`。

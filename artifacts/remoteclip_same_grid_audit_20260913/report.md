# RemoteCLIP 同尺度蒸馏审核报告

日期：2026-09-13。对象：iSAID、CLIP ViT-B/16、原版 DualFeatureMoE、RemoteCLIP 64 聚类属性库、RS 8 模板、四方向编码。

## 状态

已停止训练 PID 2557684 及其全部子进程。未启动新训练，未执行优化器更新。保留历史权重和运行配置。修改已同步至 inspur。

## 原流程发现

1. 学生使用 384×384 图像，ViT-B/16 输出 24×24、512 维稠密特征。
2. 教师按 GT 语义区域裁剪图像，再缩放至 224×224；ViT-B/32 内部网格为 7×7，调用 visual.forward 返回全局 CLS 投影向量。
3. 学生在整图特征上 ROIAlign 到 1×1，与教师裁剪图全局向量计算余弦损失。最终向量维度均为 512，但没有保证两侧空间网格相同，教师裁剪后的上下文视野也不同。
4. 因此旧实现属于区域向量蒸馏，不能描述为同尺度稠密特征蒸馏。仅将教师输入改成 384 也不够：B/32 会输出 12×12。

## 修改后的流程

- 新增 REMOTE_CLIP_DISTILL.SAME_GRID 开关，并在本次遥感模板 iSAID 配置中显式开启。
- 教师输入设为 768×768；768/32=24，学生 384/16=24。两塔均使用同一增强后的完整图像视野，教师仅调整图像分辨率，不再独立裁剪区域输入。
- 教师使用现有 encode_dense(keep_shape=True, mode='maskclip')，取末层 V 分支、残差与 MLP，再做 LN 和投影。该末层分支与当前 OpenAI 学生 dense 路径对应。
- 教师输出 [B,512,24,24]，学生 [B,576,512] 重排为同形状。没有插值教师特征图；教师位置编码使用现有实现从预训练网格插值至 24×24。
- 图像沿用 CLIP RGB 归一化，其均值、标准差与当前 open_clip 的 OpenAI 常量一致。
- 两侧均逐位置 L2 归一化，再将同一组原图区域框转换到共同 24×24 坐标。使用相同 ROIAlign 参数：output_size=1×1、aligned=True、sampling_ratio=-1；区域向量再次归一化。
- 损失仍为区域余弦损失：mean(1-cos(student_region,teacher_region))，保留 content 权重 1e-4。
- 教师 eval、requires_grad=False、no_grad、detach，且仍不注册进学生模块树；教师不进入优化器、DDP 或学生 checkpoint。
- 在编码前验证原生 patch 网格，在损失前验证完整张量形状；不匹配直接报错。

这仍是“在同尺度稠密特征图上执行相同区域池化”的蒸馏目标，不是逐像素余弦损失。语义区域选择仍按类别包围盒，最多 8 个、最小边长 8；包围盒可能包含其他类别像素，未改成掩码池化。content 分支仍监督原始 0° 学生特征，四方向编码用于分割聚合。

## 实际配置

```yaml
MODEL:
  SEM_SEG_HEAD:
    REMOTE_CLIP_DISTILL:
      SAME_GRID: true
      INPUT_SIZE: [768, 768]
      LOSS_WEIGHT: 1.0e-4
```

生效文件：configs/clip_vitb_384_isaid_attr64_dual_teacher_dual_feature_moe_rs.yaml。

其他配置保留：RS-DINO context 权重 1e-5、属性保留率 0.9、64 聚类、RS 8 提示词、四方向、全局 batch 8、学习率 2e-4、60000 步。历史配置未改写；其他旧配置 SAME_GRID 默认 false，继续保留原流程以支持复现。

## 验证证据

独立测试脚本：check_same_grid.py（远端 scripts/check_remoteclip_same_grid.py）。

- 随机空间特征图使用全图及局部框，两侧相同图产生近零损失，验证共同 ROI 坐标。
- 教师输入 224×224 与 384×384 均被网格检查拒绝。
- 在 GPU 2 加载真实学生与 RemoteCLIP 权重，单图独立特征提取：学生 [1,576,512]，教师 [1,512,24,24]。
- 实际余弦损失 0.9779502（随机输入，仅作数值有效性检查，不表示训练效果）。
- 学生特征梯度有限且非零；教师全部参数冻结且无梯度。
- optimizer_steps=0；未运行训练循环。最后 GPU 2、3 各约 409 MiB，无本次训练进程。

## 限制

已验证特征提取、形状、区域对齐及损失梯度，未执行完整训练或评估，不能据此承诺精度提升或完整 batch 的显存开销。768 输入提高教师单图 token 数；旧流程是多区域教师调用，新流程是整图调用，实际速度和峰值显存需下一次获准训练时测量。新目标与旧 CLS 区域蒸馏存在语义差异，旧日志中的 content loss 不宜直接与新目标比较。

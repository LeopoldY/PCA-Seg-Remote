# AFF 残差＋四专家

融合类型：`MODEL.SEM_SEG_HEAD.FEATURE_FUSION.TYPE=aff_residual_moe`。

## 实现

`cat_seg/modeling/transformer/aff_residual_moe.py` 提供 `AFFResidualMoE`，输入／输出均为 `[B,C,T,H,W]`，T 是类别数。

保留原双分枝 1×1 投影、四个密集专家和逐专家门控。原 `Conv1×1(2C→C)` 残差由 AFF 替换：

```text
s = spatial_proj(S), k = class_proj(K), U = concat(s, k)
A = sigmoid(local_att(s+k) + global_att(avg_pool_HW(s+k)))
Z = A*s + (1-A)*k
Y = sum_e(gate(U)[e] * expert_e(U)) + res_scale*Z
```

这里的 global_att 表示池化后的瓶颈；代码中的同名 Sequential 已包含池化，不能重复池化或跨 T 汇总。缩减比为 4，`res_scale` 初始 0.5。遵循 AFF 论文 Eq. (4) 的加权平均，没有作者部分示例代码中的额外系数 2。[Dai 等，WACV 2021](https://arxiv.org/abs/2009.14082)

专家计算按通道打包为：

```text
Conv1×1(2C→8C) → SyncBN(8C) → GELU
→ GroupConv3×3(8C→4C, groups=4C) → GELU
→ reshape(BT,4,C,H,W) → 专家加权求和
```

首层仍读取同一份 2C 输入，无需复制输入或使用 groups=4。四专家全部计算，不是 Top-k 稀疏路由。打包保留每个专家的独立参数和归一化通道统计；专家 SyncBN 调用从四次合并为一次。

AFF 两个瓶颈采用作者实现中的普通 BatchNorm2d，不新增同步通信；多卡训练时它们使用每卡统计。池化路径训练要求每卡 `B*T>1`，本项目每卡 B=4、T=15/17 满足要求。评估时使用运行统计，可处理 B=T=1。

进入卷积前显式转换到连续 NCHW 布局，兼容 Transformer 分枝返回的通道末维存储；这消除了首次真实数据 smoke 中观察到的 DDP 梯度 stride 警告。

原 FOD 的计算位置、权重和训练返回契约保持原样。旧类型 `dual_feature_moe` 仍是默认；本改动不自动转换旧 MoE checkpoint。C=128 时新模块 333,669 个参数；2 个聚合层共 4 个融合模块，即 1,334,676 个融合参数。

## 配置与启动

最终训练使用 DLRSD／iSAID 的 `*_attr64_dual_teacher_aff_residual_moe.yaml` 配置，继承各自 attr64 双教师协议。使用四方向 CLIP cost、64 属性中心、RS-DINO 上下文教师与 RemoteCLIP ViT-B/32 内容教师。两项蒸馏权重分别为 `1e-5` 和 `5e-5`，两位教师都只参与训练。

`*_attr64_rs_dino_aff_residual_moe.yaml` 与 `train_aff_residual_moe_2gpu_bs4.sh` 保留为前一阶段的单教师对照入口；最终双教师训练使用下面的新入口。

服务器项目：`/mnt/data6/yc/open-vocab/PCA-Seg-Remote`。

```bash
cd /mnt/data6/yc/open-vocab/PCA-Seg-Remote
GPU_IDS=0,1 bash scripts/train_dual_teacher_aff_residual_moe_2gpu_bs4.sh DLRSD
# 或在另一对 GPU 上训练 iSAID：
GPU_IDS=2,3 bash scripts/train_dual_teacher_aff_residual_moe_2gpu_bs4.sh iSAID
```

默认每卡 batch=4，总 batch=8，60,000 iterations，AdamW、基础学习率 0.0002、cosine 调度；完整设置以输出目录的 `config.yaml` 为准。

运行会分别写入 `output/eva_vitb_384_{dlrsd,isaid}_attr64_dual_teacher_aff_residual_moe_2gpu_bs4`，默认 `RESUME=0`。仅恢复同结构 AFF 训练时设置 `RESUME=1`。可通过 `OUTPUT_DIR`、`PYTHON_BIN`、`DATASET_ROOT` 等环境变量覆盖路径。

共享训练脚本过去会在 YAML 之后固定选择旧模块，现在支持 `FEATURE_FUSION_TYPE` 环境变量，默认值仍为旧模块；AFF 包装脚本显式设置新类型，防止新 YAML 被旧命令行默认值覆盖。

## Smoke test

```bash
cd /mnt/data6/yc/open-vocab/PCA-Seg-Remote
GPU_IDS=0,1 bash scripts/smoke_test_dual_teacher_aff_residual_moe_2gpu.sh DLRSD
GPU_IDS=2,3 bash scripts/smoke_test_dual_teacher_aff_residual_moe_2gpu.sh iSAID
```

使用各数据集固定的 8 张训练图、4 张验证图，运行 2 次优化器迭代并进行最终验证和 checkpoint 保存。默认输出与完整训练分开：`output/smoke_{dlrsd,isaid}_dual_teacher_aff_residual_moe_2gpu_bs4`。

模块级数值检查：

```bash
/opt/miniconda3/envs/yc_d2/bin/python tools/test_aff_residual_moe.py --device cpu
CUDA_VISIBLE_DEVICES=0,1 /opt/miniconda3/envs/yc_d2/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/test_aff_residual_moe.py --distributed
```

覆盖原专家与打包专家的输出／输入梯度／参数梯度／BN 统计等价性、AFF 中点及极端分枝选择、类别重排与评估时类别独立性、形状检查、checkpoint 往返、参数数目，以及 CUDA 混合精度和 DDP 优化器更新。

实际服务器结果另见 `AFF_RESIDUAL_MOE_SMOKE_RESULTS.md`。Smoke 指标仅用于验证数据和计算流程，不能作为收敛精度或性能加速证据。

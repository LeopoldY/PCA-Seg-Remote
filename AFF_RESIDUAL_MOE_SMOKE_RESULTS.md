# AFF 残差＋四专家：inspur 验证记录

2026-09-06 完成单教师对照验证，2026-09-07 按用户追加要求完成双教师组合验证。最终代码同步至 `/mnt/data6/yc/open-vocab/PCA-Seg-Remote`。

## 最终双教师组合验证（2026-09-07）

RS-DINO＋RemoteCLIP 双教师、AFF 残差＋四专家、64 属性中心，DLRSD 在 GPU 0–1、iSAID 在 GPU 2–3 同时进行双卡 smoke。两项均完成 2 次训练迭代、4 张图验证、checkpoint 保存并退出 0。

| 数据集 | total_loss | RS-DINO 加权损失 | RemoteCLIP 加权损失 | 验证 mIoU |
|---|---:|---:|---:|---:|
| DLRSD | 0.683636 | 4.969807e-5 | 5.334193e-5 | 8.042585 |
| iSAID | 0.586035 | 5.085614e-5 | 5.453281e-5 | 2.626078 |

两个蒸馏开关均实际启用，权重为 `1e-5`／`5e-5`；两项教师损失均非零且有限。四个 AFF 融合位置均完成两次 BN 更新，checkpoint 的 iteration=1。最终日志无梯度 stride 警告或 Traceback；教师参数不进入学生 checkpoint。12 个代码／配置／测试文件与本地 SHA-256 一致。

服务器产物：

```text
/mnt/data6/yc/open-vocab/PCA-Seg-Remote/output/smoke_dlrsd_dual_teacher_aff_residual_moe_2gpu_bs4/
/mnt/data6/yc/open-vocab/PCA-Seg-Remote/output/smoke_isaid_dual_teacher_aff_residual_moe_2gpu_bs4/
```

本地日志：[DLRSD 双教师](artifacts/aff_residual_moe_smoke_20260906/smoke_dlrsd_dual_teacher.log)、[iSAID 双教师](artifacts/aff_residual_moe_smoke_20260906/smoke_isaid_dual_teacher.log)、[双教师配置与 checkpoint 核验](artifacts/aff_residual_moe_smoke_20260906/verified_dual_teacher_summary.json)。两步 smoke 的精度不用于评判融合方案效果。

## 前一阶段：模块与单教师对照验证

以下保留 2026-09-06 的模块测试和单教师对照记录。

环境：Python 3.9.25，PyTorch 2.8.0+cu128，CUDA 12.8，Detectron2 0.6，NVIDIA GeForce RTX 4090。使用 attr64＋四方向 CLIP cost＋训练期 RS-DINO 教师协议；仅替换融合模块。

## 模块检查

CPU 和双卡 CUDA 测试均退出 0，全部检查通过：

- 原专家与打包专家的前向、输入梯度、参数梯度、BN 运行统计在容差内一致。
- AFF 权重为中点、趋近 0／1 时分别对应预期的分枝加权。
- 输入输出契约、类别重排、评估时类别间独立性、参数计数和 checkpoint 往返通过。
- 双卡 DDP、FP32／AMP 训练均产生有限梯度，优化器更新后各卡参数一致。
- 使用实际 Transformer 风格的非连续输入进行梯度测试。

日志：[CPU](artifacts/aff_residual_moe_smoke_20260906/unit_cpu_final.log)、[双卡 CUDA](artifacts/aff_residual_moe_smoke_20260906/unit_ddp_cuda_final.log)。

## 真实数据 smoke

每个数据集取 8 张训练图、4 张验证图；每卡 batch=4，总 batch=8；运行 2 次优化器迭代，完成验证并保存 `model_final.pth`。两个进程均退出 0。

| 数据集 | GPU | 日志记录 total_loss | 验证 mIoU | 结果 |
|---|---|---:|---:|---|
| DLRSD | 0、1 | 1.964810 | 7.417958 | 通过 |
| iSAID | 2、3 | 1.935531 | 2.414251 | 通过 |

以上 mIoU 是两步训练后在 4 张验证图上的值，不代表收敛精度。某些逐类别指标为 NaN，是 smoke 子集中缺失对应类别导致；整体 mIoU 与训练损失均为有限值。

已从最终配置和 checkpoint 额外确认：

1. 实际 `FEATURE_FUSION.TYPE=aff_residual_moe`、`MAX_ITER=2`、总 batch=8。
2. 四个融合位置均包含 `packed_experts`；四个专家归一化计数均为 2。
3. checkpoint 的零起始 iteration 为 1，即完成两次迭代。
4. 所有融合浮点状态为有限值，四个 `res_scale` 均从初始 0.5 得到更新。
5. 最终两份日志均无 DDP 梯度 stride 警告，也没有 Traceback。

最初一轮 smoke 发现 DDP 梯度布局警告；随后显式规范融合输入的卷积布局，复跑模块检查及两个完整 smoke。首次输出保留，表中只引用最终版结果。初始化随机种子由原训练配置生成，两轮指标不用于比较该布局修改的精度影响。

日志仍有原依赖的 timm 弃用／可选 apex 提示，以及 Detectron2 启动器退出时的 NCCL process-group 清理提示；测试正常退出。未修改这些上游依赖。

## 服务器产物

完整 smoke checkpoint 与 `config.yaml`、`metrics.json`、验证产物位于：

```text
/mnt/data6/yc/open-vocab/PCA-Seg-Remote/output/smoke_dlrsd_aff_residual_moe_2gpu_bs4_final_20260906/
/mnt/data6/yc/open-vocab/PCA-Seg-Remote/output/smoke_isaid_aff_residual_moe_2gpu_bs4_final_20260906/
```

服务器核验日志目录：

```text
/mnt/data6/yc/open-vocab/PCA-Seg-Remote/output/aff_residual_moe_validation_20260906/
```

本地保留 [DLRSD 完整日志](artifacts/aff_residual_moe_smoke_20260906/smoke_dlrsd_final.log)、[iSAID 完整日志](artifacts/aff_residual_moe_smoke_20260906/smoke_isaid_final.log) 和 [配置／checkpoint／SHA-256 核验结果](artifacts/aff_residual_moe_smoke_20260906/verified_summary.json)。大体积 checkpoint 保留在服务器。

被修改的两个服务器原文件已备份在 `backups/aff_residual_moe_20260906/`：`model.py` 和共享双卡训练脚本。新 AFF 模块采用独立类型，旧模型默认构造路径保持原样。

## 启动完整训练

按用户最终要求，以下命令在 inspur 上执行，使用双教师蒸馏＋AFF 四专家＋64 属性，分别使用两对 GPU 在后台训练；本次任务仅运行 smoke，未启动这两项完整训练。

```bash
cd /mnt/data6/yc/open-vocab/PCA-Seg-Remote
mkdir -p output/launch_logs

GPU_IDS=0,1 OMP_NUM_THREADS=2 nohup bash scripts/train_dual_teacher_aff_residual_moe_2gpu_bs4.sh DLRSD \
  > output/launch_logs/dual_teacher_aff_dlrsd.log 2>&1 &

GPU_IDS=2,3 OMP_NUM_THREADS=2 nohup bash scripts/train_dual_teacher_aff_residual_moe_2gpu_bs4.sh iSAID \
  > output/launch_logs/dual_teacher_aff_isaid.log 2>&1 &
```

默认训练 60,000 iterations、每项训练总 batch=8、AdamW、基础学习率 0.0002。RS-DINO／RemoteCLIP 损失权重分别为 `1e-5`／`5e-5`。输出分别为 `output/eva_vitb_384_dlrsd_attr64_dual_teacher_aff_residual_moe_2gpu_bs4` 与对应的 `isaid` 目录。

若要恢复同一个 AFF 训练，在相应命令前增加 `RESUME=1`；旧 DualFeatureMoE checkpoint 不应直接当作此新结构的断点恢复。

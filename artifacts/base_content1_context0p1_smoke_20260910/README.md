# ViT-B 双教师权重 smoke test（2026-09-10）

已将 OpenAI CLIP ViT-B/16 + AFFResidualMoE 的 DLRSD、iSAID 配置设为 content=1.0、context=0.1。ViT-L 配置显式保留原权重，避免继承变化。配置及训练脚本已同步至 inspur。

测试主机 inspur；GPU 4、5（RTX 4090），启动前均为 15 MiB / 0% 利用率。每组 8 张训练图、4 张验证图，global batch=8，2 次参数更新；两组退出码均为 0，完成验证与 model_final.pth 保存。未启动正式训练。

下表为 metrics.json 在 iteration=1 写出的平滑损失（两步窗口），所有分项损失均已乘以表头所列权重。organ 的 0.001 在模型内部应用于各层、每层两次正交损失的平均值；total 为加权分项之和，无额外权重。

| Dataset | sem_seg ×1 | organ ×0.001 | content ×1 | context ×0.1 | total（加权和） |
|---|---:|---:|---:|---:|---:|
| DLRSD | 0.681207433 | 0.000034193 | 1.011327788 | 0.608676374 | 2.301245788 |
| iSAID | 0.706061587 | 0.000075265 | 1.032727689 | 0.698911279 | 2.437775820 |

原始 context 损失可由表中数值除以 0.1 得到。DLRSD/iSAID 的 smoke mIoU 为 2.251489 / 6.033506，仅用于验证流程。验证集中缺失类别的指标含 NaN，训练损失无 NaN。峰值显存分别为 22786 / 21777 MiB。

两组均出现进程退出时未 destroy_process_group 的 NCCL 清理警告，iSAID 还出现 TCPStore 提前关闭警告；两组退出码均为 0，指标及 checkpoint 已成功写入。完整 stdout/stderr 保存在 dlrsd.log 和 isaid.log。

训练脚本：scripts/train_rskt_clip_dual_teacher_aff_residual_moe_2gpu.sh。默认 60000 iterations，global batch=8，初始学习率 0.0002。新输出目录包含 content1_context0p1，避免覆盖旧实验。以下命令仅提供，未执行；运行前重新确认 GPU 空闲。

```bash
cd /mnt/data6/yc/open-vocab/PCA-Seg-Remote
GPU_IDS=4,5 RESUME=0 bash scripts/train_rskt_clip_dual_teacher_aff_residual_moe_2gpu.sh DLRSD
GPU_IDS=6,7 RESUME=0 bash scripts/train_rskt_clip_dual_teacher_aff_residual_moe_2gpu.sh iSAID
```

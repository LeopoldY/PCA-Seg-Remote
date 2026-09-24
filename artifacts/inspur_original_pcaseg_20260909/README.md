# inspur 原版 PCA-Seg / OpenAI ViT-B/16 遥感训练

服务器项目：`/mnt/data6/yc/open-vocab/PCA-Seg`

原项目完整归档：`/mnt/data6/yc/open-vocab/archives/PCA-Seg_20260909_140515`（约 114 GB，包含旧代码、权重和输出；采用同盘目录移动，未删除文件）。归档旁保存原提交号和 tracked diff。

官方来源：https://github.com/PixelSegTech/PCA-Seg.git

新克隆提交：`59c90eaa79ebb83906c65e9a857e118f38c7397d`。克隆完成时 git status 为空，之后仅增加本适配及下述必要修复。

## 配置与数据

继承官方 `configs/vitb_384.yaml` 与 `configs/config.yaml`：OpenAI CLIP ViT-B/16、384×384 训练输入、PCA/EPL/FOD、attention 微调、全局 batch 4、AdamW、基础 LR 0.0002、CLIP LR 倍率 0.01、weight decay 0.0001、cosine、80,000 iterations、每 5,000 iterations 验证，验证短边 640，关闭滑窗。未改变官方损失定义（包括其对 255 像素的现有 BCE 行为）。

每个数据集独立模型，各两卡，每卡 batch 2。分割头随机初始化，CLIP 从官方原始预训练权重初始化。

权重：`/home/yc/.cache/clip/ViT-B-16.pt`，由官方 CLIP 加载器读取并校验。
SHA256：`5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`。
该值与官方加载器 URL 中的校验值一致。`MODEL.WEIGHTS` 留空，不加载旧任务模型。

环境：`/opt/miniconda3/envs/yc_d2/bin/python`，Python 3.9 / PyTorch 2.8.0+cu128；复用现有可用环境，与官方建议的 Python 3.10 / PyTorch 2.0 不完全相同。

数据根：`/mnt/data6/yc/datasets/OVSISBenchDataset`，可用 `OVSISBENCH_DATASETS` 覆盖。

| 数据集 | 训练 | 验证 | 类别 | 路径 |
|---|---:|---:|---:|---|
| DLRSD | 5601 | 1401 | 17 | `DLRSD_split/{train,val}/{imgs,D2masks}` |
| iSAID | 18076 | 6363 | 15 | `iSAID_split/{train,val}/{images,D2masks}` |

全部图像与掩码配对、尺寸、标签值已检查；两划分文件名交集为零。沿用已有划分，未做源影像级或内容哈希去重。DLRSD 标签 0–16，iSAID 标签 0–14 / 255，验证忽略 255。完整统计位于 `output/rs_data_audit.json`。

## 必要源码修改

1. `train_net.py` 尊重数据路径环境变量。
2. 注册独立的遥感训练/验证及 smoke 子集。
3. `cat_seg_model.py` 训练分支向分割头补传 `batched_inputs`，修复实测 TypeError。
4. `transformer/model.py` 将 `need_loss` 作为关键字传入聚合层，恢复官方 FOD 损失计算；CCA 诊断依赖改为函数内懒加载。
5. `cat_seg_predictor.py` 训练时清理验证文本缓存，确保后续验证使用更新后的 CLIP 文本特征。

没有迁入旧项目的属性融合、额外教师或其他实验模块。可用 `git diff` 检查全部 tracked 修改。

## 启动

登录 inspur 后：

```bash
cd /mnt/data6/yc/open-vocab/PCA-Seg
# 双数据集后台并行；自动创建带时间戳输出目录并打印路径和 PID
DLRSD_GPUS=0,1 ISAID_GPUS=2,3 bash scripts/launch_rs_parallel.sh
```

单数据集：

```bash
GPU_IDS=0,1 bash scripts/train_rs_original.sh DLRSD
GPU_IDS=2,3 bash scripts/train_rs_original.sh iSAID
```

重新启动时不要与仍在运行的训练争用 GPU。并行启动器支持固定 `RUN_ROOT`；单任务支持 `OUTPUT_DIR`。断点续训：

```bash
RESUME=1 RUN_ROOT=/absolute/path/to/existing_run \
  DLRSD_GPUS=0,1 ISAID_GPUS=2,3 bash scripts/launch_rs_parallel.sh
```

独立验证：

```bash
GPU_IDS=0,1 EVAL_ONLY=1 \
  WEIGHTS=/absolute/path/to/DLRSD/model_final.pth \
  OUTPUT_DIR=/absolute/path/to/eval_dlrsd \
  bash scripts/train_rs_original.sh DLRSD
```

iSAID 同理。评估完整 val 划分，输出 mIoU、fwIoU、mACC、pACC 和逐类结果。配置尾部可以增加 Detectron2 overrides。

两次迭代的快速训练/验证：

```bash
SMOKE=1 RUN_ROOT=/absolute/path/to/new_smoke_run \
  bash scripts/launch_rs_parallel.sh
```

此模式使用每数据集 8 张训练图和 4 张验证图，仅用于运行检查，指标不代表最终性能。

## 本次已启动的正式任务

2026-09-09 14:12（服务器时间）启动，14:13 检查两个任务均已完成 60 iterations，loss 有限且下降。

- RUN_ROOT：`/mnt/data6/yc/open-vocab/PCA-Seg/output/original_openai_vitb16_parallel_20260909`
- DLRSD：GPU 0,1，主进程 PID 1596202，iter 59 total_loss 0.1814。
- iSAID：GPU 2,3，主进程 PID 1596203，iter 59 total_loss 0.07924。
- 当前状态：训练中，尚未完成 80k；首次全量验证在 iter 5000。
- 两个 smoke 任务均完成 2 iterations、每轮验证和 model_final.pth 保存；DLRSD 独立加载/验证退出码为 0。smoke 子集缺失类别会产生逐类 NaN，非训练损失 NaN。
- 当前 Detectron2 / PyTorch 组合在分布式退出时会输出未显式销毁进程组的警告；短训练与独立评估已完成。

查看日志：

```bash
tail -f /mnt/data6/yc/open-vocab/PCA-Seg/output/original_openai_vitb16_parallel_20260909/DLRSD.log
tail -f /mnt/data6/yc/open-vocab/PCA-Seg/output/original_openai_vitb16_parallel_20260909/iSAID.log
```

本地交付目录包含新增适配文件、`adaptation/upstream_fixes.patch`、数据审计及 smoke 指标。若复建：从上述提交的全新官方仓库复制新增文件，然后运行 `scripts/apply_upstream_fixes.py` 和 `scripts/fix_training_call.py`；或仅应用保存的 patch（二者不要重复执行）。

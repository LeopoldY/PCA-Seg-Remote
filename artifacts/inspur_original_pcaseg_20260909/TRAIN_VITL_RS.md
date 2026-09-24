# 原版 PCA-Seg Large 遥感训练

服务器项目：`/mnt/data6/yc/open-vocab/PCA-Seg`。

脚本：`scripts/train_rs_original_vitl.sh`、`scripts/launch_rs_vitl_parallel.sh`。
配置：`configs/vitl_336_dlrsd_rs.yaml`、`configs/vitl_336_isaid_rs.yaml`，继承官方 `vitl_336.yaml`。

初始 CLIP 权重：`/mnt/data6/yc/open-vocab/PCA-Seg-Remote/pretrained/ViT-L-14-336px.pt`。
SHA256：`3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`，与官方一致。

通过 `MODEL.SEM_SEG_HEAD.CACHE_DIR` 指定本地 OpenAI 权重，加载器已添加可选本地路径及 SHA256 校验；架构标识仍为 `ViT-L/14@336px`。`MODEL.WEIGHTS` 为空，分割部分随机初始化，不加载 Base 或其他训练模型。沿用此前修复的官方参数传递和验证缓存问题。

每任务 2 GPU，每 GPU batch 2，全局 batch 4；AdamW、LR 0.0002、CLIP multiplier 0.01、cosine、80,000 iterations、每 5,000 iterations 验证。保留官方 384×384 裁剪，在模型内部缩放到 CLIP 336×336。未加入属性库、教师或其他实验模块。

DLRSD 使用既有 train/val 划分（5601/1401、17 类），iSAID 使用既有 train/val 划分（18076/6363、15 类）。

## 并行启动

```bash
cd /mnt/data6/yc/open-vocab/PCA-Seg
DLRSD_GPUS=0,1 ISAID_GPUS=2,3 bash scripts/launch_rs_vitl_parallel.sh
```

替换 GPU 编号即可，必须是互不重叠的两组双卡。脚本通过 nohup 后台启动两项任务，无需额外添加 `&`。
默认输出：`output/original_openai_vitl336_parallel_时间戳/{DLRSD,iSAID}/`，父目录中保存对应 `.log` 和 `.pid`。

## 单数据集

```bash
GPU_IDS=0,1 bash scripts/train_rs_original_vitl.sh DLRSD
GPU_IDS=2,3 bash scripts/train_rs_original_vitl.sh iSAID
```

## 续训

```bash
RESUME=1 RUN_ROOT=/绝对路径/既有并行输出目录 \
  DLRSD_GPUS=0,1 ISAID_GPUS=2,3 bash scripts/launch_rs_vitl_parallel.sh
```

仅检查命令、不启动：

```bash
CHECK_ONLY=1 DLRSD_GPUS=0,1 ISAID_GPUS=2,3 bash scripts/launch_rs_vitl_parallel.sh
```

验证记录：shell 语法、配置合并、并行启动参数检查通过；在 CPU 上通过原版 CLIP 构建接口成功加载用户指定权重，确认 24 个视觉 Transformer blocks、14×14 patch、577×1024 位置嵌入。尚未进行 Large GPU 前向/反向或启动训练，未测量实际峰值显存。

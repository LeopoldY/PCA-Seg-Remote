# PCA-Seg Large 最佳权重：八数据集验证

服务器项目：`/mnt/data6/yc/open-vocab/PCA-Seg`。
脚本已部署，语法、权重存在性、八数据集配对及 Large 配置一致性已检查，未启动验证或 GPU 推理。

入口：`scripts/eval_rs8_vitl_best_parallel.sh`，实现：`scripts/eval_rs8_vitl_best.py`。

默认训练目录：`output/original_openai_vitl336_parallel_20260910_152746`。
按各自训练时验证集 mIoU 选择最佳权重：

- DLRSD：`DLRSD/model_final.pth`，mIoU 92.130345。
- iSAID：`iSAID/model_0034999.pth`，mIoU 86.314896。

```bash
cd /mnt/data6/yc/open-vocab/PCA-Seg
nohup bash scripts/eval_rs8_vitl_best_parallel.sh 0 1 \
  > eval_rs8_vitl_best.launch.log 2>&1 &
```

0、1 分别为两个权重使用的物理 GPU，可替换为两张不同 GPU。两个权重并行，各自顺序评估 DLRSD、iSAID、Potsdam、Vaihingen、UDD5、LoveDA、UAVid、VDD，合计 16 组结果。各目标使用独立进程，加载 Large 配置并切换目标类别词表。

默认使用八数据集全量口径，DLRSD/iSAID 包含训练样本。若这两个目标仅测试 val，追加 `--source-scope val`，其他六个目标范围不变。

结果：`output/eval_rs8_vitl_best_时间戳/summary.csv`。同目录 `manifest.json` 记录权重、GPU 和数据范围，每个 `<source>/<target>/` 保存独立日志和 `result.json`。

可追加 `--output-root /绝对路径/新目录` 或 `--run-root /绝对路径/训练父目录`。为防止覆盖，输出目录必须不存在。

仅检查、不推理：

```bash
bash scripts/eval_rs8_vitl_best_parallel.sh 0 1 --check-only
```

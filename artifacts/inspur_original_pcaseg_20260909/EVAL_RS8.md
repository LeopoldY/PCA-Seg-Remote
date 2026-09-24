# 最佳权重并行评估 8 个遥感数据集

服务器：`/mnt/data6/yc/open-vocab/PCA-Seg`。仅已部署脚本并进行语法、配对与掩码抽查，未启动验证。

入口 `scripts/eval_rs8_best_parallel.sh`，实现 `scripts/eval_rs8_best.py`。两个位置参数分别指定 DLRSD 权重和 iSAID 权重使用的物理 GPU，每个权重顺序评估 8 个数据集。每个目标数据集启动独立进程，避免文本特征缓存或类别词表串用。保留官方输入尺寸、池化、滑窗等配置。

按训练时各自验证集 mIoU 自动选择最佳已保存 checkpoint：

- DLRSD：`output/original_openai_vitb16_parallel_20260909/DLRSD/model_final.pth`，mIoU 91.590694。
- iSAID：`output/original_openai_vitb16_parallel_20260909/iSAID/model_0054999.pth`，mIoU 82.477666。

```bash
cd /mnt/data6/yc/open-vocab/PCA-Seg
nohup bash scripts/eval_rs8_best_parallel.sh 0 1 > eval_rs8_best.launch.log 2>&1 &
```

把 `0 1` 替换为指定的两张不同 GPU。默认后台启动命令只在用户执行时运行。

默认沿用已有全量 8 数据集口径：DLRSD 7002、iSAID 24439、Potsdam 20102、Vaihingen 2254、UDD5 160（train+val）、LoveDA 2522、UAVid 270、VDD 400。DLRSD/iSAID 全量包含训练样本，不代表独立验证集指标。若这两个目标只测 val：

```bash
nohup bash scripts/eval_rs8_best_parallel.sh 0 1 --source-scope val > eval_rs8_best_val.launch.log 2>&1 &
```

其余 6 个数据集保持已有全量口径。Potsdam 显式映射 `_RGB_` 到 `_label_`，UAVid 映射 `_Images_` 到 `_Labels_`；Vaihingen 只使用存在对应图像的掩码，跳过额外的 1026 张孤立掩码。

输出默认为 `output/eval_rs8_best_时间戳/`，含 `manifest.json`（权重、GPU 和数据范围）、`summary.csv`（16 组结果）、每个源权重的 `summary.json`，以及 `<source>/<target>/{console.log,result.json,inference/}`。任何目标失败会继续处理其他目标，最终非零退出并在汇总中标记失败。

可追加 `--output-root /绝对路径/新目录` 或 `--run-root /绝对路径/训练父目录`。输出目录必须不存在，防止覆盖。只检查文件而不运行模型：

```bash
bash scripts/eval_rs8_best_parallel.sh 0 1 --check-only
```

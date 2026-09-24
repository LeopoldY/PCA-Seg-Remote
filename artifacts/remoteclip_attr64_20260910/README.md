# RemoteCLIP 属性库编码（2026-09-10）

已在 inspur 使用现有 RemoteCLIP 权重、CPU float32 完成 DLRSD 和 iSAID 属性描述编码，结果同步回本地。

| 文本编码器 | 属性中心矩阵 | DLRSD 类别标记 | iSAID 类别标记 |
|---|---|---|---|
| RemoteCLIP ViT-B-32 | 512 × 64 | 17 × 64 | 15 × 64 |
| RemoteCLIP ViT-L-14 | 768 × 64 | 17 × 64 | 15 × 64 |

输入为 attributes_text/{DLRSD,iSAID}_train_descriptors.json。保持既有处理方式：文本转小写、文本嵌入 L2 归一化、KMeans(random_state=0) 聚成 64 类，保存 [cluster_bank, class_flags]。聚类中心保持现有格式，不额外归一化。

使用 tools/build_attribute_database.py 的 open_clip 后端直接加载 RemoteCLIP checkpoint 并调用 encode_text；修复了该后端创建图像 transforms 时依赖训练 args 的问题。

四个属性库均通过项目原生加载器检查，并验证维度、有限值、非零中心、二值类别标记和类别/聚类覆盖。本地文件 SHA256 与服务器一致。详细来源与哈希见 verification.json，运行日志见 build_b32.log、build_l14.log。

更新了 ViT-B/L 的四份 AFFResidualMoE 配置及对应训练入口，默认使用新库；新训练输出目录包含 remoteclip_attr，原属性库保留。历史评估脚本仍保留旧库默认值，评估新模型时须用 ATTRIBUTE_DATABASE 指定本次生成的对应属性库。

只更换属性库的文本编码来源，未更换学生的类别文本编码器。未启动分割训练。

重建命令：

```bash
bash scripts/build_rskt_clip_attr64.sh
bash scripts/build_rskt_clipl_attr64.sh
```

默认 CPU，支持 DEVICE、REMOTECLIP_CHECKPOINT、PYTHON_BIN 和 FORCE=1。服务器旧配置备份为 previous_configs.tar，旧 Python 构建器为 build_attribute_database.previous.py。

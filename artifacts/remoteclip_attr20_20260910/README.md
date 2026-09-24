# RemoteCLIP 20 中心属性库（2026-09-10）

使用 inspur 上现有 RemoteCLIP B/32、L/14 权重重新编码 DLRSD、iSAID 属性描述，通过既有 ExCEL KMeans(random_state=0) 流程全库聚类为 20 个属性中心。

- B/32：中心矩阵 [512,20]。
- L/14：中心矩阵 [768,20]。
- DLRSD 类别标记 [17,20]；iSAID 类别标记 [15,20]。

输入为 attributes_text/{DLRSD,iSAID}_train_descriptors.json。既有构建流程对句子转小写、编码、L2 归一化后进行聚类；不对旧的 64 中心再次聚类。

输出为 attributes_text/rskt_seg/{dataset}_train_desc_remoteclip_{b32,l14}_cluster_20_embedding_bank.pth。保留原 64 中心库和配置。

新增四份 clip_vit{b,l}_*_attr20_dual_teacher_aff_residual_moe.yaml 配置，明确设置 ATTR_FUSION.NUM_CLUSTERS=20 和对应库路径。ViT-B/L 双 GPU 训练入口及 iSAID ViT-B 单 GPU 入口默认切换到 attr20，输出目录也改为 attr20；未启动分割训练。

重建：

```bash
bash scripts/build_rskt_clip_attr20.sh
bash scripts/build_rskt_clipl_attr20.sh
```

默认 CPU，FORCE=1 可重新生成。verify.py 校验矩阵维度、有限值、非零中心、二值类别标记、类别和中心覆盖，以及 Detectron2 配置完整继承后的属性维度匹配。verification.json 记录文件哈希与输入来源，build_b32.log/build_l14.log 为编码日志。

旧训练入口已在服务器备份为 previous_launchers.tar。历史评估脚本默认仍面向旧实验；评估新模型须同时选用 attr20 配置和 cluster_20 库。

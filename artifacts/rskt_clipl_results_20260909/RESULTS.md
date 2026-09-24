# ViT-L 训练结果统计（2026-09-09）

配置：OpenAI CLIP ViT-L/14@336px、64 中心属性库、RS-DINO + RemoteCLIP-L 双教师蒸馏、aff_residual_moe。
来源：inspur 上对应训练输出的 metrics.json、config.yaml、log.txt。以下为各自验证集结果，不是跨数据集测试结果。
两套均完成 60,000 步，保存 model_final.pth 并完成最终验证。所有指标单位为 %。

| 数据集 | checkpoint | 完成步数 | mIoU | fwIoU | mACC | pACC |
|---|---|---:|---:|---:|---:|---:|
| DLRSD | model_0054999.pth | 55000 | 92.3485 | 94.2505 | 95.8662 | 97.0105 |
| DLRSD | model_final.pth | 60000 | 92.3442 | 94.2440 | 95.8711 | 97.0069 |
| iSAID | model_0029999.pth | 30000 | 85.9191 | 89.3433 | 93.2067 | 94.2860 |
| iSAID | model_final.pth | 60000 | 85.3530 | 88.6114 | 92.6265 | 93.8731 |

DLRSD 最佳与最终仅差 0.0043 个百分点；iSAID 最终较最佳下降 0.5661 个百分点。按验证 mIoU 选择模型时，分别使用上述最佳 checkpoint。

## 验证曲线数据

| 完成步数 | DLRSD mIoU | iSAID mIoU |
|---:|---:|---:|
| 5000 | 82.3598 | 85.1476 |
| 10000 | 86.9501 | 85.5919 |
| 15000 | 89.0039 | 85.3362 |
| 20000 | 90.0335 | 85.0349 |
| 25000 | 90.8885 | 85.5909 |
| 30000 | 91.4345 | 85.9191 |
| 35000 | 91.6827 | 85.0344 |
| 40000 | 92.0157 | 85.6124 |
| 45000 | 92.2287 | 85.5799 |
| 50000 | 92.3060 | 85.4619 |
| 55000 | 92.3485 | 85.4168 |
| 60000 | 92.3442 | 85.3530 |

## DLRSD 逐类 IoU

| 类别 | 最佳 checkpoint | 最终 checkpoint |
|---|---:|---:|
| airplane | 88.1332 | 88.1140 |
| bare soil | 91.5748 | 91.5577 |
| buildings | 96.1029 | 96.1000 |
| cars | 87.3229 | 87.3236 |
| chaparral | 75.8657 | 75.8777 |
| court | 97.9285 | 97.9335 |
| dock | 82.5727 | 82.6031 |
| field | 98.3016 | 98.2699 |
| grass | 94.8486 | 94.8431 |
| mobile home | 91.7265 | 91.6978 |
| pavement | 96.2513 | 96.2524 |
| sand | 94.0713 | 94.0435 |
| sea | 99.5497 | 99.5497 |
| ship | 89.3876 | 89.4155 |
| tanks | 97.6732 | 97.6720 |
| trees | 93.6527 | 93.6284 |
| water | 94.9609 | 94.9689 |

训练计时（日志原文，不含之后的最终评估）：17:21:33 (0:21:39 on hooks)。
最后记录训练 loss：0.006325；已检查所有记录的训练 loss 均为有限值。

服务器模型目录：`/mnt/data6/yc/open-vocab/PCA-Seg-Remote/output/clip_vitl_336_dlrsd_attr64_dual_teacher_aff_residual_moe_2gpu_bs4/`。

## iSAID 逐类 IoU

| 类别 | 最佳 checkpoint | 最终 checkpoint |
|---|---:|---:|
| baseball diamond | 91.1401 | 88.4955 |
| basketball court | 78.5114 | 77.9949 |
| bridge | 76.2136 | 78.0111 |
| ground track field | 77.1608 | 75.1892 |
| harbor | 85.5367 | 84.7896 |
| helicopter | 64.7131 | 65.1004 |
| large vehicle | 92.0257 | 91.3524 |
| plane | 99.1757 | 99.2324 |
| roundabout | 85.4233 | 87.4277 |
| ship | 88.8738 | 88.6352 |
| small vehicle | 88.9527 | 88.1651 |
| soccer ball field | 88.2140 | 86.8820 |
| storage tank | 97.4083 | 97.6396 |
| swimming pool | 79.3390 | 75.4459 |
| tennis court | 96.0989 | 95.9346 |

训练计时（日志原文，不含之后的最终评估）：16:54:41 (1:02:42 on hooks)。
最后记录训练 loss：0.004083；已检查所有记录的训练 loss 均为有限值。

服务器模型目录：`/mnt/data6/yc/open-vocab/PCA-Seg-Remote/output/clip_vitl_336_isaid_attr64_dual_teacher_aff_residual_moe_2gpu_bs4/`。

中间日志 iteration 为零基索引，例如 54999 对应完成 55,000 步；最终评估记录为 iteration=60000。
本次仅统计已有训练和验证记录，没有重新运行评估。

# CLIP ViT-L/14@336px + attributes + dual teachers + AFFResidualMoE

Student: OpenAI CLIP ViT-L/14@336px, dense 336px / 24x24 grid, four rotations.
Input crop remains 384px. Text and appearance guidance are 768-dimensional;
intermediate visual guidance uses the existing 1024-dimensional L architecture.
Attribute banks: 64 centers, rebuilt separately for DLRSD and iSAID using their
training descriptors and the exact student text encoder.
Context teacher: existing RS-DINO RSIB.pth, Q-relation loss weight 1e-5.
Content teacher: RemoteCLIP ViT-L-14, cosine loss weight 5e-5. The B/32 teacher
has 512-dimensional output and cannot be used with the L student's 768-dimensional
content loss. Teacher crop size remains 224px; teacher microbatch is 8.
Both teachers are frozen and excluded from segmentation checkpoints.

Official weight sources:
- https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt
- https://huggingface.co/chendelong/RemoteCLIP/blob/main/RemoteCLIP-ViT-L-14.pt

Both files were downloaded locally before upload. See weights.sha256 for official
SHA-256 values. RS-DINO reuses the existing server checkpoint.

## Full training commands (server: inspur)

```bash
cd /mnt/data6/yc/open-vocab/PCA-Seg-Remote
GPU_IDS=0,1 bash scripts/train_rskt_clipl_dual_teacher_aff_residual_moe_2gpu.sh DLRSD
# In another terminal, if training both datasets concurrently:
GPU_IDS=2,3 bash scripts/train_rskt_clipl_dual_teacher_aff_residual_moe_2gpu.sh iSAID
```

60,000 iterations, global batch 8, gradient accumulation 1, learning rate 2e-4,
CLIP multiplier 0.01, evaluation/checkpoint every 5,000 iterations.
Output: output/clip_vitl_336_{dlrsd,isaid}_attr64_dual_teacher_aff_residual_moe_2gpu_bs4.
RESUME=1 resumes only the corresponding new ViT-L training checkpoint.

## Rebuild banks and repeat smoke tests

```bash
bash scripts/build_rskt_clipl_attr64.sh
GPU_IDS=0,1 bash scripts/smoke_test_rskt_clipl_dual_teacher_aff_residual_moe_2gpu.sh DLRSD
GPU_IDS=2,3 bash scripts/smoke_test_rskt_clipl_dual_teacher_aff_residual_moe_2gpu.sh iSAID
```

## Verified results on inspur, 2026-09-08

Both local and uploaded weights passed their official SHA-256 checks.
Both banks were rebuilt successfully: cluster_bank=(768,64), class_flags=(17,64)
for DLRSD and (15,64) for iSAID.

Each smoke run used 8 training images, 4 validation images, 2 optimizer steps,
and 2 GPUs with global batch 8. Both exited with code 0, wrote model_final.pth,
and completed validation.

| Dataset | Total loss | Context loss | Content loss | Peak allocated GPU memory |
|---|---:|---:|---:|---:|
| DLRSD | 0.7069786 | 6.8936e-5 | 1.7288e-5 | 33624 MiB |
| iSAID | 0.6251387 | 8.0193e-5 | 1.6641e-5 | 32602 MiB |

Smoke mIoU: DLRSD 1.6662; iSAID 13.6175. These are pipeline checks after two
steps, not trained model performance. NaN per-class metrics reflect classes
absent from the four validation images. Existing optional-apex, timm deprecation,
and distributed process-group cleanup warnings did not prevent success.

The fresh DLRSD checkpoint loaded with strict=True and produced finite
[17,384,384] inference output with both teacher checkpoint paths nonexistent and
both teacher objects absent. The checkpoint contains no teacher weights.

RSKT native attention vs PCA SDPA max absolute difference: 0.0001859665;
atol=rtol=2e-4 passed. The earlier 1e-4 comparison failed on 6/443136 elements,
also with TF32 disabled. Source inspection identified need_weights=True vs False;
using the same attention kernel gave max difference exactly 0.0. The training
implementation was not changed to alter its attention kernel.

Full 60,000-step training has not been started. Use the commands above.


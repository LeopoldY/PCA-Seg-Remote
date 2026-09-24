# OpenAI CLIP ViT-B/16 adaptation, 2026-09-07

The backbone follows RSKT-Seg's vitb_384_DLRSD/iSAID configurations:
OpenAI ViT-B/16 with its original pretrained checkpoint, 384px dense encoding,
and four rotations. RSKT-Seg also provides a separate L/14 variant; this
experiment uses B/16 to retain the existing 384px, 24x24 feature-grid setup.

Training retains the current dual teachers (RS-DINO context and RemoteCLIP
content), AFFResidualMoE, 64 attribute centers, and 60000 iterations.
Attribute banks were rebuilt on inspur with the OpenAI text encoder from the
same training-only descriptors. EVA attribute banks are not reused.

The final OpenAI block keeps RSKT's dense V output. A temporary ln_1 hook
provides its normalized input for Q/K projections used by context distillation.
It is removed after each forward; inference does not install the hook.

## Verified on inspur

- DLRSD: GPUs 0,1, 8 training images, 4 validation images, 2 optimizer steps,
  global batch 8. Exit code 0; checkpoint and validation completed.
  Total loss 0.761898; context loss 0.0000604045; content loss 0.0000503239.
- iSAID: GPUs 2,3, same scope. Exit code 0; checkpoint and validation completed.
  Total loss 0.647047; context loss 0.0000702376; content loss 0.0000518309.
- Dense-feature parity with the server's RSKT-Seg model_vpt.py:
  max absolute difference 0.00006103515625, within atol=rtol=1e-4.
- Fresh DLRSD checkpoint loading passed strict=True. Inference produced finite
  [17,384,384] output with both teacher checkpoint paths nonexistent and both
  teacher objects absent.
- Existing distributed teardown emitted NCCL process-group cleanup warnings;
  both runs exited successfully after validation.

The smoke mIoU values (DLRSD 2.7993, iSAID 6.1483) are only pipeline checks
after two training steps, not model performance measurements. NaN per-class
metrics correspond to classes absent from the four-image validation subset.
See DLRSD/metrics.json, DLRSD/log.txt and iSAID equivalents.

## Full training

From /mnt/data6/yc/open-vocab/PCA-Seg-Remote, use separate terminals:

```bash
GPU_IDS=0,1 bash scripts/train_rskt_clip_dual_teacher_aff_residual_moe_2gpu.sh DLRSD
GPU_IDS=2,3 bash scripts/train_rskt_clip_dual_teacher_aff_residual_moe_2gpu.sh iSAID
```

Outputs use output/clip_vitb_384_{dlrsd,isaid}_attr64_dual_teacher_aff_residual_moe_2gpu_bs4.
Start these experiments afresh; old EVA checkpoints use a different architecture.
RESUME=1 is supported for checkpoints from these new output directories.

The corresponding configs are configs/clip_vitb_384_{dlrsd,isaid}_attr64_dual_teacher_aff_residual_moe.yaml.
The default pretrained path is /mnt/data6/yc/open-vocab/RSKT-Seg/pretrained/ViT-B-16.pt;
override CLIP_CHECKPOINT if moving the project.
To generate the attribute banks on another host, run bash scripts/build_rskt_clip_attr64.sh.
To repeat smoke tests, run bash scripts/smoke_test_rskt_clip_dual_teacher_aff_residual_moe_2gpu.sh DLRSD
(or iSAID), with GPU_IDS set to two available devices.

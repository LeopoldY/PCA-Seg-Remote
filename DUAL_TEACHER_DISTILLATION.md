# DeCLIP-style dual-teacher backbone distillation

During segmentation training the EVA02-CLIP-B/16 student is initialized from
the COCO-trained DeCLIP checkpoint and receives two complementary backbone
objectives:

1. **RS-DINO context:** match the final-layer 24x24 patch self-similarity
   matrix with the student's final attention `q` (or `q+k`) similarity.
2. **RemoteCLIP content:** form class-region boxes from each semantic target,
   encode their 224x224 image crops with frozen RemoteCLIP ViT-B/32, and align
   those global crop embeddings with ROI-pooled student dense features using
   the DeCLIP cosine content loss.

The teachers are loaded lazily only on the first training forward. They are
kept outside the PyTorch module tree, never enter DDP/optimizer/state dicts,
and are absent during inference. Losses are reported separately as
`loss_rs_dino_context` and `loss_remote_clip_content`.

The reference RSKT-Seg weights are expected at:

- `pretrained/RSIB.pth`
- `pretrained/RemoteCLIP-ViT-B-32.pt`

The student initialization is expected at:

- `pretrained/declip_eva_b16_dinov2b_coco_full_retry1_epoch_latest_state_dict.pt`

Use `scripts/train_dual_teacher_rskt_protocol_4gpu_bs2.sh` for full training
and `scripts/smoke_test_dual_teacher_rskt_protocol_1gpu.sh` for validation.

The full-training launcher defaults to GPUs `0,1,2,3`, two images per GPU,
and a global batch size of eight.

The distillation weights are calibrated from the DLRSD smoke-test loss scale:
RS-DINO context uses `1.0e-5` and RemoteCLIP content uses `5.0e-5`. Their
weighted losses are therefore approximately `5e-5`, matching the order of
magnitude of the internally weighted `loss_organ`.

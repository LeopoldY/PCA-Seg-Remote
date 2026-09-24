# RS-DINO dual-backbone adaptation

This adaptation keeps PCA-Seg's text-side attribute enrichment and all PCA
parallel aggregation blocks, EPL fusion, FOD loss, training loss, inference,
and sliding-window behavior.  Only the image/cost path and the two decoder
guidance inputs are extended from one CLIP backbone to CLIP + RS-DINO.

## Data flow

1. PCA-Seg obtains CLIP dense features and CLIP multi-scale guidance exactly
   as before.
2. The same normalized input is resized to 384 x 384 and encoded by the
   RSKT-Seg-compatible DINO ViT-B/8 (`RSIB.pth`).
3. The DINO final patch feature is projected from 768 channels and downsampled
   from 48 x 48 to PCA-Seg's 24 x 24 text dimension.  Layers 4 and 8 provide
   48 x 48 and 96 x 96 decoder guidance.
4. `AttributeTextAdapter` enriches CLIP text features once.  Both CLIP and
   RS-DINO then use that exact enriched tensor to construct normalized cosine
   cost volumes.
5. Following RSKT-Seg's `simple_separate` fusion, the two costs use independent
   7 x 7 embeddings and sigmoid gates.  Their concatenation is fused by a
   7 x 7 convolution and sigmoid, then receives the RSKT CLIP-cost residual.
6. The fused volume enters the unchanged PCA-Seg parallel aggregation stack.
   CLIP and DINO guidance are concatenated during the two existing upsampling
   stages using RSKT-Seg's channel layout.

## Configuration

Use one of the opt-in configurations:

- `configs/eva_vitb_384_attr_fusion_rs_dino.yaml`
- `configs/eva_vitl_336_attr_fusion_rs_dino.yaml`

Place the RSKT-Seg RSIB checkpoint at `pretrained/RSIB.pth`, or override:

```bash
MODEL.SEM_SEG_HEAD.RS_DINO.WEIGHTS /absolute/path/to/RSIB.pth
```

Supported `FINETUNE` values match the reference implementation:
`frozen` (default), `attention`, and `full`.  Existing configurations have
`RS_DINO.ENABLED: false` by default and retain the pre-adaptation code path.

## Verification scope

This checkout has no runnable training environment.  The adaptation is
therefore limited to source/configuration integration and static validation;
no training, checkpoint conversion, or metric claim is included.

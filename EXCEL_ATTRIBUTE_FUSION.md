# ExCEL attribute fusion in PCA-Seg

This extension ports only ExCEL's Text Semantic Enrichment (TSE) into the
official PCA-Seg aggregation path.

For every PCA-Seg class-prompt vector `t` and ExCEL cluster bank `A`, it uses
the operation from ExCEL's `model/load_attr.py`:

```text
logits = t @ A
weights = softmax(mask_lowest_10_percent(logits))
enriched_t = normalize(t + weights @ A.T)
```

PCA-Seg then uses `enriched_t` in the same cost-volume construction and text
guidance positions where it used `t`. The batch and prompt axes are flattened
only to apply the same ExCEL operation independently; no visual retrieval,
confidence gate, spatial injection, or additional trainable fusion parameter
is introduced.

Set `MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED false` (the default in all original
configs) to bypass the adapter. This leaves the official PCA-Seg forward path
and checkpoint parameter shapes unchanged.

## Database construction

The official JSON and OpenAI CLIP ViT-B/16 database files are stored under
`attributes_text/excel/`. For EVA02-CLIP-B-16, rebuild the same database with:

```bash
bash scripts/build_excel_attribute_databases.sh
```

This produces:

- VOC: `[512, 112]` center bank and `[20, 112]` class flags;
- COCO: `[512, 224]` center bank and `[80, 224]` class flags.

## Run and rollback

Use `configs/eva_vitb_384_attr_fusion.yaml` for the COCO database or
`configs/eva_vitb_384_attr_fusion_voc.yaml` for the VOC database. To run the
unaltered method with the same command, append:

```text
MODEL.SEM_SEG_HEAD.ATTR_FUSION.ENABLED false
```

# ExCEL attribute databases

This directory vendors the attribute files from `zwyang6/ExCEL` commit
`2cd7f510d62793c76365268a2cc4da7cbe915e4d`.

Official source files:

| file | SHA-256 |
| --- | --- |
| `descriptors_pascal_voc_gpt4.0_cluster_a_photo_of4.json` | `43f1feaee48872893a320f14f02ddad0aa2d65e3af08a57d62491a532dbb5e65` |
| `descriptors_ms_coco_gpt4.0_cluster_a_photo_of4.json` | `2d0f65c4dff4fd27b76846a829b794c5a14b9104260f1fbdc2c2e1616787fc31` |
| `pascal_voc_desc_clip_ViT-B-16_gpt4.0_cluster_112_embedding_bank.pth` | `d394c49d901e7a5b5ef1ea324366647b356d6ab873e41e7b8d8dcd03819fa5d1` |
| `ms_coco_desc_clip_ViT-B-16_gpt4.0_cluster_224_embedding_bank.pth` | `1b2d607598a5245a78caceda4a85002a7781f0292b507f6bb8c85bbdc88384b5` |

The two official `.pth` files use OpenAI CLIP ViT-B/16. PCA-Seg's EVA
variants must rebuild the same ExCEL database from the unchanged official
JSON text with their own text encoder:

```bash
pip install -r requirements-excel.txt
bash scripts/build_excel_attribute_databases.sh
```

The builder intentionally retains ExCEL's procedure and serialization:

1. lowercase every official descriptor;
2. encode and L2-normalize each descriptor;
3. run `sklearn.cluster.KMeans(n_clusters=..., random_state=0)` globally;
4. save `[cluster_bank, class_flags]`, with shapes `[D, M]` and `[C, M]`.

No descriptors, attribute types, image-conditioned retrieval, extra
filtering, or alternative clustering are added.

EVA02-CLIP-B/16 construction results (built on `inspur` with the PCA-Seg
checkpoint `pretrained/EVA02_CLIP_B_psz16_s8B.pt`):

| file | shapes | SHA-256 |
| --- | --- | --- |
| `pascal_voc_desc_eva02_clip_b16_gpt4.0_cluster_112_embedding_bank.pth` | `[512,112]`, `[20,112]` | `8424d18843cbcc2617c5c1505a77acbc8c7b71448bf0c03b5b14cb1e9b56486c` |
| `ms_coco_desc_eva02_clip_b16_gpt4.0_cluster_224_embedding_bank.pth` | `[512,224]`, `[80,224]` | `4994c69ef04dc8bb8a8d719db6463a064fdb64f3c5374a881be4be0cd33c994f` |

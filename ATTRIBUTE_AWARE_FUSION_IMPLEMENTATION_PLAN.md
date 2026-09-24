# Attribute fusion implementation note

The earlier custom image-conditioned attribute retrieval and spatial gating
design has been retired. It was not the ExCEL attribute fusion method.

The active implementation now ports only ExCEL Text Semantic Enrichment (TSE)
onto the official PCA-Seg text/cost-volume path. See
[`EXCEL_ATTRIBUTE_FUSION.md`](EXCEL_ATTRIBUTE_FUSION.md) for the exact method,
database provenance, build commands, validation, and fallback switch.

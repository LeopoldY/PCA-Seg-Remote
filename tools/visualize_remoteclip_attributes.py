#!/usr/bin/env python3
"""Encode original descriptor strings and plot individual embeddings by class."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import hashlib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from build_attribute_database import load_text_encoder

OUT = ROOT / 'artifacts/remoteclip_raw_attributes_20260910'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
results = []
for tag, model_name in [('b32', 'ViT-B-32'), ('l14', 'ViT-L-14')]:
    model, tokenizer = load_text_encoder('open_clip', model_name, str(ROOT / 'pretrained' / f'RemoteCLIP-{model_name}.pt'), torch.device('cpu'))
    for dataset in ['DLRSD', 'iSAID']:
        source = ROOT / 'attributes_text' / f'{dataset}_train_descriptors.json'
        descriptions = json.loads(source.read_text())
        classes = list(descriptions)
        texts = [text for values in descriptions.values() for text in values]
        labels = np.array([i for i, values in enumerate(descriptions.values()) for _ in values])
        with torch.no_grad():
            embeddings = torch.cat([torch.nn.functional.normalize(model.encode_text(tokenizer(texts[i:i+128])).float(), dim=-1) for i in range(0, len(texts), 128)])
        assert torch.isfinite(embeddings).all()
        assert torch.allclose(embeddings.norm(dim=-1), torch.ones(len(texts)), atol=1e-5)
        stem = f'{dataset}_remoteclip_{tag}_raw'
        torch.save(dict(embeddings=embeddings, labels=torch.from_numpy(labels), classes=classes, texts=texts), OUT / f'{stem}.pth')
        values = embeddings.numpy()
        pca = PCA(n_components=2, svd_solver='full')
        xy_pca = pca.fit_transform(values)
        # Unit vectors: Euclidean distances are monotonic in cosine distance.
        xy_tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42).fit_transform(values)
        result = dict(dataset=dataset, model=model_name, tag=tag, count=len(texts), dim=values.shape[1], classes=classes,
                      texts=texts, labels=labels.tolist(), pca=xy_pca.tolist(), tsne=xy_tsne.tolist(),
                      pca_explained_variance=pca.explained_variance_ratio_.tolist(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        results.append(result)
        print(stem, values.shape, 'encoded and projected', flush=True)
    del model

(OUT/'projections.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print('All embeddings and projections saved.',flush=True)

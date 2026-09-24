#!/usr/bin/env python3
"""Train-only classwise attribute mixup; manifest first, then RemoteCLIP encoding.

ProxyDet-inspired text-only adaptation, not the detector's multimodal loss.
No target vocabulary is read until all expanded embedding banks are saved.
"""
import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = [
    'a photo of a {} in the scene', 'a remote sensing photo of {}',
    'a land cover scene including {}', 'an aerial view of {}',
    'a satallite image of {}', 'fields of {}',
    'a landscape covered with {}', '{} visible from above',
]
DATASETS = ['DLRSD', 'iSAID', 'LoveDA', 'Potsdam', 'Vaihingen', 'UDD5', 'VDD', 'uavid']

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def manifest(dataset, seed, samples):
    source = ROOT / 'attributes_text' / (dataset + '_train_descriptors.json')
    original = json.loads(source.read_text())
    entries = []
    for cls, texts in original.items():
        prefix = 'an aerial image of ' + cls + '.'
        for i, text in enumerate(texts):
            if not text.startswith(prefix):
                raise ValueError('Unexpected descriptor prefix: ' + text)
            suffix = text[len(prefix):].strip()
            entries.append(dict(id=f'{cls}:{i}', source_class=cls, original_text=text,
                                prompts=[t.format(cls) + '. ' + suffix for t in TEMPLATES]))
    rng = random.Random(seed)
    proxies = []
    for a, b in itertools.combinations(original, 2):
        for _ in range(samples):
            proxies.append(dict(id=f'proxy_{len(proxies):04d}', parent_classes=[a, b],
                                weight_a=rng.betavariate(1, 1),
                                operation='l2_normalize(lambda * prototype_a + (1-lambda) * prototype_b)'))
    return dict(schema_version=1, dataset=dataset, source_sha256=sha(source),
                source_file=str(source.relative_to(ROOT)), seed=seed, beta=1.0,
                samples_per_unordered_pair=samples, original_descriptors=original,
                original_entries=entries, proxies=proxies, templates=TEMPLATES,
                template_source='/mnt/data6/yc/open-vocab/RSKT-Seg/RSKT_Seg/third_party/imagenet_templates.py:RS_TEMPLATES',
                prototype_rule='L2-normalized mean of template-ensembled attribute vectors within each training class',
                adaptation='Static all-distinct-class-pairs sampling; text-only attribute prototypes. No generated prose or target classes.')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encode', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--samples-per-pair', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--checkpoint-dir', type=Path, default=ROOT/'pretrained')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/proxy_rskt_20260913')
    args = p.parse_args()
    assert args.samples_per_pair > 0
    banks = []
    for dataset in DATASETS[:2]:
        bank = manifest(dataset, args.seed, args.samples_per_pair)
        dest = ROOT/'attributes_text/proxy_rskt'/f'{dataset}_expanded.json'
        dump(dest, bank)
        assert bank['original_descriptors'] == json.loads((ROOT/bank['source_file']).read_text())
        banks.append(bank)
        print(dataset, len(bank['original_entries']), 'original +', len(bank['proxies']), 'proxy recipes saved', flush=True)
    if not args.encode:
        return

    import numpy as np
    import torch
    import torch.nn.functional as F
    import open_clip
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for tag, architecture in [('b32', 'ViT-B-32'), ('l14', 'ViT-L-14')]:
        checkpoint = args.checkpoint_dir/f'RemoteCLIP-{architecture}.pt'
        model = open_clip.create_model(architecture, pretrained=str(checkpoint), device=args.device).float().eval()
        tokenizer = open_clip.get_tokenizer(architecture)

        @torch.inference_mode()
        def encode(groups):
            flat = list(itertools.chain.from_iterable(groups))
            vectors = []
            for start in range(0, len(flat), 128):
                tokens = tokenizer(flat[start:start+128]).to(args.device)
                vectors.append(F.normalize(model.encode_text(tokens).float(), dim=-1).cpu())
            features = torch.cat(vectors).reshape(len(groups), len(TEMPLATES), -1)
            return F.normalize(features.mean(1), dim=-1)

        encoded = []
        for bank in banks:
            classes = list(bank['original_descriptors'])
            original = encode([x['prompts'] for x in bank['original_entries']])
            labels = torch.tensor([classes.index(x['source_class']) for x in bank['original_entries']])
            prototypes = F.normalize(torch.stack([original[labels == i].mean(0) for i in range(len(classes))]), dim=-1)
            proxies = F.normalize(torch.stack([
                r['weight_a'] * prototypes[classes.index(r['parent_classes'][0])] +
                (1-r['weight_a']) * prototypes[classes.index(r['parent_classes'][1])]
                for r in bank['proxies']]), dim=-1)
            expanded = torch.cat([original, proxies])
            assert torch.isfinite(expanded).all()
            assert torch.allclose(expanded.norm(dim=1), torch.ones(len(expanded)), atol=1e-5)
            assert torch.equal(expanded[:len(original)], original)
            payload = dict(embeddings=expanded, original_embeddings=original, proxy_embeddings=proxies,
                           class_prototypes=prototypes, original_labels=labels, classes=classes,
                           original_count=len(original), metadata=bank, checkpoint_sha256=sha(checkpoint),
                           architecture=architecture, torch_version=str(torch.__version__))
            stem = bank['dataset'] + '_remoteclip_' + tag
            torch.save(payload, args.output/(stem + '_expanded.pth'))
            # Existing ExCEL loader accepts this unclustered [D,M], [C,M] bank.
            # Proxy membership indicates its two source classes; lambda is in metadata.
            flags = torch.zeros(len(classes), len(expanded))
            flags[labels, torch.arange(len(original))] = 1
            for i, r in enumerate(bank['proxies']):
                for cls in r['parent_classes']:
                    flags[classes.index(cls), len(original)+i] = 1
            torch.save([expanded.T.contiguous(), flags], args.output/(stem + '_embedding_bank.pth'))
            encoded.append((bank, original, prototypes, proxies))
            print(stem, tuple(expanded.shape), 'expanded bank saved', flush=True)

        # Read comparison-only datasets AFTER saving expanded banks.
        vocab = {d: json.loads((ROOT/'datasets'/f'{d}.json').read_text()) for d in DATASETS}
        excluded = {'no-data', 'background', 'clutter', 'unlabeled', 'ignore',
                    'clutter/background', 'background clutter', 'other'}
        rows = [dict(dataset=d, name=n) for d, names in vocab.items() for n in names if n.lower() not in excluded]
        unique = list(dict.fromkeys(r['name'] for r in rows))
        targets = encode([[t.format(name) for t in TEMPLATES] for name in unique])
        torch.save(dict(embeddings=targets, names=unique, rows=rows, templates=TEMPLATES), args.output/f'target_classes_{tag}.pth')
        for bank, original, prototypes, proxies in encoded:
            base = torch.cat([original, proxies, prototypes]).numpy()
            joint = np.concatenate([base, targets.numpy()])
            pca = PCA(n_components=2, svd_solver='full').fit(base)
            pca_xy = pca.transform(joint)
            tsne_xy = TSNE(n_components=2, perplexity=30, init='pca', learning_rate=200,
                           random_state=args.seed).fit_transform(joint)
            cos_orig = targets @ original.T
            cos_proto = targets @ prototypes.T
            cos_proxy = targets @ proxies.T
            comparison = []
            for row in rows:
                if row['dataset'] == bank['dataset']:
                    continue
                k = unique.index(row['name'])
                val, idx = cos_proxy[k].max(0)
                comparison.append(dict(**row, exact_train_name_overlap=row['name'].lower() in [c.lower() for c in bank['original_descriptors']],
                    best_original_cos=float(cos_orig[k].max()), best_prototype_cos=float(cos_proto[k].max()),
                    best_proxy_cos=float(val), proxy_id=bank['proxies'][int(idx)]['id'],
                    parents=bank['proxies'][int(idx)]['parent_classes'],
                    weight_a=bank['proxies'][int(idx)]['weight_a'],
                    expanded_gain=float(max(cos_orig[k].max(), val)-cos_orig[k].max())))
            result = dict(dataset=bank['dataset'], model=architecture, tag=tag,
                          original_count=len(original), proxy_count=len(proxies), classes=list(bank['original_descriptors']),
                          target_names=unique, target_rows=rows, pca=pca_xy.tolist(), tsne=tsne_xy.tolist(),
                          pca_variance=pca.explained_variance_ratio_.tolist(), comparisons=comparison)
            results.append(result)
            dump(args.output/'projections.json', results)
            print(bank['dataset'], tag, 'comparison and projections saved', flush=True)
        del model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

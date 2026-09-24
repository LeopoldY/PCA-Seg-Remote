import hashlib
import json
from pathlib import Path
import torch
from cat_seg.modeling.attribute_fusion import load_excel_attribute_database

root = Path.cwd()
results = []
for tag, dim, model in [('b32', 512, 'ViT-B-32'), ('l14', 768, 'ViT-L-14')]:
    checkpoint = root / 'pretrained' / f'RemoteCLIP-{model}.pt'
    for dataset in ['DLRSD', 'iSAID']:
        descriptors = root / 'attributes_text' / f'{dataset}_train_descriptors.json'
        descriptions = json.loads(descriptors.read_text())
        path = root / 'attributes_text' / 'rskt_seg' / f'{dataset}_train_desc_remoteclip_{tag}_cluster_64_embedding_bank.pth'
        bank, flags = load_excel_attribute_database(str(path), expected_dim=dim, expected_num_clusters=64)
        assert flags.shape == (len(descriptions), 64)
        assert torch.isfinite(bank).all() and torch.isfinite(flags).all()
        assert (bank.norm(dim=0) > 0).all()
        assert ((flags == 0) | (flags == 1)).all()
        assert (flags.sum(dim=0) > 0).all() and (flags.sum(dim=1) > 0).all()
        result = dict(dataset=dataset, encoder=f'RemoteCLIP-{model}', checkpoint=str(checkpoint.resolve()),
                      descriptors_sha256=hashlib.sha256(descriptors.read_bytes()).hexdigest(),
                      descriptor_count=sum(map(len, descriptions.values())), classes=list(descriptions),
                      bank=str(path.relative_to(root)), bank_shape=list(bank.shape), flags_shape=list(flags.shape),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(), finite=True)
        results.append(result)
        print(dataset, model, tuple(bank.shape), tuple(flags.shape), 'PASS')
(root / 'artifacts/remoteclip_attr64_20260910/verification.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')

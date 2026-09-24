"""Verify all native masks, image pairing, dimensions and split disjointness."""
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

root=Path(os.getenv('OVSISBENCH_DATASETS','/mnt/data6/yc/datasets/OVSISBenchDataset'))
report={}
for dataset,folder,ext,n in [('DLRSD','imgs','jpg',17),('iSAID','images','png',15)]:
    entry={}
    ids={}
    for split in ('train','val'):
        base=root/(dataset+'_split')/split
        images={p.stem:p for p in (base/folder).glob('*.'+ext)}
        masks={p.stem:p for p in (base/'D2masks').glob('*.png')}
        assert images and images.keys()==masks.keys(), (dataset,split,'unpaired')
        ids[split]=set(images)
        def check(k):
            with Image.open(masks[k]) as f: a=np.asarray(f)
            with Image.open(images[k]) as f: size=f.size
            assert a.ndim==2 and a.shape==size[::-1], (k,a.shape,size)
            values,counts=np.unique(a,return_counts=True)
            assert set(values.tolist()) <= set(range(n))|{255}, (k,values)
            return Counter(dict(zip(values.tolist(),counts.tolist())))
        counts=Counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for c in pool.map(check, sorted(images)): counts.update(c)
        entry[split]={'samples':len(images),'pixels_by_label':dict(sorted(counts.items()))}
        print(dataset,split,entry[split],flush=True)
    assert not ids['train'] & ids['val'], (dataset,'split overlap')
    entry['train_val_filename_overlap']=0
    report[dataset]=entry
Path('output').mkdir(exist_ok=True)
Path('output/rs_data_audit.json').write_text(json.dumps(report,indent=2))

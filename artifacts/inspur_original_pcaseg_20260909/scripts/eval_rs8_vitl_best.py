#!/usr/bin/env python3
"""Large PCA-Seg: two best source checkpoints, one GPU each, fresh process per target."""
import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['DLRSD', 'iSAID', 'Potsdam', 'Vaihingen', 'UDD5', 'LoveDA', 'uavid', 'VDD']
SPECS = {
    'DLRSD': [('DLRSD/imgs', 'DLRSD/D2masks', 'jpg')],
    'iSAID': [('iSAID/imgs', 'iSAID/D2masks', 'png')],
    'Potsdam': [('Potsdam/imgs', 'Potsdam/D2masks', 'png')],
    'Vaihingen': [('Vaihingen/imgs', 'Vaihingen/D2masks', 'png')],
    'UDD5': [('UDD5/train/src', 'UDD5/train/gt', 'JPG'), ('UDD5/val/src', 'UDD5/val/gt', 'JPG')],
    'LoveDA': [('LoveDA/images_png', 'LoveDA/masks_png', 'png')],
    'uavid': [('uavid/Images', 'uavid/Labels', 'png')],
    'VDD': [('VDD/src', 'VDD/gt', 'JPG')],
}


def get_records(target, data_root, scope):
    specs = SPECS[target]
    if scope == 'val' and target in ('DLRSD', 'iSAID'):
        folder, ext = ('imgs', 'jpg') if target == 'DLRSD' else ('images', 'png')
        specs = [(f'{target}_split/val/{folder}', f'{target}_split/val/D2masks', ext)]
    records = []
    for imgs, masks, ext in specs:
        images = {p.stem:p for p in (data_root/imgs).glob('*.'+ext)}
        labels = {p.stem:p for p in (data_root/masks).glob('*.png')}
        if not images:
            raise ValueError(f'Empty image directory: {target}: {imgs}')
        # Match explicit naming conventions; never pair by sorted position.
        for k in sorted(images):
            key = k.replace('_RGB_', '_label_') if target == 'Potsdam' else k
            if target == 'uavid': key = k.replace('_Images_', '_Labels_')
            if key not in labels:
                raise ValueError(f'Missing mask for {images[k]}: expected {key}.png')
            records.append({'file_name':str(images[k]), 'sem_seg_file_name':str(labels[key])})
        # Vaihingen has extra orphan masks. Only evaluate its actual images.
    return records


def select_best(run_root, source):
    directory = run_root/source
    rows = [json.loads(line) for line in (directory/'metrics.json').read_text().splitlines() if line.strip()]
    rows = [r for r in rows if isinstance(r.get('sem_seg/mIoU'), (int,float)) and math.isfinite(r['sem_seg/mIoU'])]
    if not rows:
        raise ValueError(f'No validation mIoU: {directory}')
    # Detectron2's final evaluation may be logged at MAX_ITER rather than MAX_ITER-1.
    best = max(rows, key=lambda r:(r['sem_seg/mIoU'], -r['iteration']))
    iteration = int(best['iteration'])
    checkpoint = directory/f'model_{iteration:07d}.pth'
    if iteration == 80000:
        checkpoint = directory/'model_final.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Best checkpoint missing; no fallback to a worse one: {checkpoint}')
    return {'source':source, 'iteration':iteration, 'val_mIoU':best['sem_seg/mIoU'], 'checkpoint':str(checkpoint)}


def evaluate_one(args):
    # Import torch/Detectron2 only in the child after CUDA_VISIBLE_DEVICES is set.
    sys.path.insert(0,str(ROOT))
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.engine import default_argument_parser
    from train_net import main
    classes = json.loads((ROOT/'datasets'/f'{args.target}.json').read_text())
    records = get_records(args.target, args.data_root, args.source_scope)
    name = f'rs8_{args.target}_{args.source_scope}_sem_seg'
    DatasetCatalog.register(name, lambda:records)
    MetadataCatalog.get(name).set(stuff_classes=classes,
        stuff_dataset_id_to_contiguous_id={i:i for i in range(len(classes))},
        evaluator_type='sem_seg', ignore_label=255)
    parsed = default_argument_parser().parse_args([
        '--config-file', f'configs/vitl_336_{args.source.lower()}_rs.yaml', '--eval-only',
        'MODEL.WEIGHTS', str(args.checkpoint),
        'MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON', f'datasets/{args.target}.json',
        'MODEL.SEM_SEG_HEAD.NUM_CLASSES', str(len(classes)),
        'DATASETS.TEST', repr((name,)), 'OUTPUT_DIR', str(args.output_root),
        'DATALOADER.NUM_WORKERS', str(args.workers),
    ])
    results = main(parsed)
    def clean(value):
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if hasattr(value,'item'): value=value.item()
        if isinstance(value,float) and not math.isfinite(value): return None
        return value
    (args.output_root/'result.json').write_text(json.dumps(clean(results),indent=2,allow_nan=False))


def run_source(source, gpu, selected, args):
    rows=[]
    for target in TARGETS:
        out=args.output_root/source/target
        out.mkdir(parents=True,exist_ok=False)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--evaluate-one',
             '--source',source,'--target',target,'--checkpoint',selected['checkpoint'],
             '--output-root',str(out),'--data-root',str(args.data_root),
             '--source-scope',args.source_scope,'--workers',str(args.workers)]
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS=env.get('OMP_NUM_THREADS','4'),
                   OVSISBENCH_DATASETS=str(args.data_root))
        print(f'[{source} GPU {gpu}] {target} -> {out}',flush=True)
        with (out/'console.log').open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        row={'source':source,'target':target,'GPU':gpu,'checkpoint':selected['checkpoint'],
             'status':'ok' if result.returncode==0 else f'failed:{result.returncode}'}
        if result.returncode==0:
            metrics=json.loads((out/'result.json').read_text())['sem_seg']
            row.update({key:metrics.get(key) for key in ['mIoU','fwIoU','mACC','pACC']})
        rows.append(row)
        (args.output_root/source/'summary.json').write_text(json.dumps(rows,indent=2))
    return rows


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',nargs=2,metavar=('DLRSD_GPU','ISAID_GPU'))
    p.add_argument('--run-root',type=Path,default=ROOT/'output/original_openai_vitl336_parallel_20260910_152746')
    p.add_argument('--output-root',type=Path)
    p.add_argument('--data-root',type=Path,default=Path(os.getenv('OVSISBENCH_DATASETS','/mnt/data6/yc/datasets/OVSISBenchDataset')))
    p.add_argument('--source-scope',choices=['all','val'],default='all',help='DLRSD/iSAID target split. all includes training data; other six datasets retain existing all-data protocol.')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--check-only',action='store_true',help='Only inspect checkpoints, JSONs and dataset pairing; no model loading or inference.')
    p.add_argument('--evaluate-one',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--source',choices=['DLRSD','iSAID'],help=argparse.SUPPRESS)
    p.add_argument('--target',choices=TARGETS,help=argparse.SUPPRESS)
    p.add_argument('--checkpoint',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    os.chdir(ROOT)
    args.data_root=args.data_root.resolve()
    if args.evaluate_one:
        evaluate_one(args)
        return
    if not args.gpus or len(set(args.gpus))!=2 or not all(g.isdigit() for g in args.gpus):
        p.error('--gpus requires two different physical GPU indices, e.g. --gpus 0 1')
    args.run_root=args.run_root.resolve()
    selected={s:select_best(args.run_root,s) for s in ['DLRSD','iSAID']}
    counts={}
    for t in TARGETS:
        classes=json.loads((ROOT/'datasets'/f'{t}.json').read_text())
        if not classes or not all(isinstance(c,str) for c in classes): raise ValueError(t)
        counts[t]={'images':len(get_records(t,args.data_root,args.source_scope)),'classes':len(classes)}
    manifest={'checkpoints':selected,'datasets':counts,'source_scope':args.source_scope,
              'gpu_assignment':dict(zip(['DLRSD','iSAID'],args.gpus))}
    print(json.dumps(manifest,indent=2),flush=True)
    if args.check_only: return
    args.output_root=(args.output_root or ROOT/'output'/('eval_rs8_vitl_best_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    args.output_root.mkdir(parents=True,exist_ok=False)
    (args.output_root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_source,s,g,selected[s],args) for s,g in zip(['DLRSD','iSAID'],args.gpus)]
        rows=[row for f in futures for row in f.result()]
    with (args.output_root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['source','target','GPU','checkpoint','status','mIoU','fwIoU','mACC','pACC'])
        w.writeheader(); w.writerows(rows)
    print(f'Summary: {args.output_root}/summary.csv',flush=True)
    if any(r['status']!='ok' for r in rows): raise SystemExit(1)

if __name__=='__main__': cli()

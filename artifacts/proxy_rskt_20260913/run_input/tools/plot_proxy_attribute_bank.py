#!/usr/bin/env python3
"""Plot joint projections and full-dimensional cosine comparisons."""
import argparse
import csv
import json
import textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'artifacts/proxy_rskt_20260913')
    args = p.parse_args()
    out = args.output
    results = json.loads((out/'projections.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    stats = []
    for r in results:
        stem = r['dataset'] + '_remoteclip_' + r['tag']
        comparisons = r['comparisons']
        with (out/(stem+'_cosine_comparison.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        others = list(dict.fromkeys(row['dataset'] for row in comparisons))
        n, m, c = r['original_count'], r['proxy_count'], len(r['classes'])
        for method in ['pca', 'tsne']:
            xy = np.asarray(r[method])
            fig, axes = plt.subplots(2, 4, figsize=(22, 13))
            for ax, ds in zip(axes.flat, others):
                ax.scatter(*xy[:n].T, s=9, c='#8796a5', alpha=.3, label='Original attributes')
                ax.scatter(*xy[n:n+m].T, s=10, c='#ee9b27', alpha=.4, label='Proxy classes')
                ax.scatter(*xy[n+m:n+m+c].T, s=45, c='#1e5790', marker='s', edgecolors='white', linewidths=.4, label='Train attribute prototypes')
                names = [row['name'] for row in comparisons if row['dataset']==ds]
                ids = [n+m+c+r['target_names'].index(name) for name in names]
                ax.scatter(*xy[ids].T, s=100, c='#bd2042', marker='*', edgecolors='white', linewidths=.4, label='Comparison class names', zorder=5)
                placed = []
                scale = np.ptp(xy, axis=0)
                for k, (name, idx) in enumerate(zip(names, ids)):
                    origin = xy[idx] / scale
                    candidates = [origin + radius*np.array([np.cos(angle),np.sin(angle)])
                                  for radius in [.025,.05,.075,.1,.125,.15]
                                  for angle in np.linspace(0,2*np.pi,12,endpoint=False)]
                    pos = next((v for v in candidates if all(np.linalg.norm(v-z)>.042 for z in placed)), candidates[-1])
                    placed.append(pos)
                    ax.annotate(str(k+1), xy[idx], xytext=pos*scale, fontsize=8,
                                arrowprops=dict(arrowstyle='-', color='#9e6470', lw=.45), zorder=6)
                ax.set_title(ds + '\n' + textwrap.fill('; '.join(f'{i+1}: {v}' for i,v in enumerate(names)), width=52), fontsize=8)
                ax.grid(alpha=.12)
                ax.set_xlabel(method.upper()+' 1'); ax.set_ylabel(method.upper()+' 2')
                ax.set_xlim(xy[:,0].min()-.05*np.ptp(xy[:,0]), xy[:,0].max()+.05*np.ptp(xy[:,0]))
                ax.set_ylim(xy[:,1].min()-.05*np.ptp(xy[:,1]), xy[:,1].max()+.05*np.ptp(xy[:,1]))
            axes.flat[-1].axis('off')
            handles, labels = axes.flat[0].get_legend_handles_labels()
            axes.flat[-1].legend(handles, labels, loc='upper left', frameon=False, markerscale=1.4)
            note = ('PCA fit on train attributes + proxies + prototypes;\ncomparison names projected into the same basis.\nExplained variance: '+', '.join(f'{v:.1%}' for v in r['pca_variance']) if method=='pca' else 'One joint t-SNE fit, shared across all seven panels.\nPerplexity 30; seed 42.\nOnly local neighborhoods are interpretable.')
            axes.flat[-1].text(.02,.53,note+'\n\nComparison classes include shared names and synonyms;\nthey are not all strictly unseen classes.\nNo comparison vocabulary was used to build proxies.\n\nTemplates: RSKT-Seg RS_TEMPLATES (8).', va='top', transform=axes.flat[-1].transAxes, linespacing=1.6)
            fig.suptitle(f"{r['dataset']} train-only proxies | RemoteCLIP {r['model']} | {method.upper()}\n{n} original attributes + {m} proxy vectors", fontsize=17)
            fig.tight_layout(rect=(0,0,1,.93), h_pad=3)
            fig.savefig(out/(stem+'_'+method+'.png'), dpi=160)
            fig.savefig(out/(stem+'_'+method+'.pdf'))
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, max(9, len(comparisons)*.21)))
        matrix = np.array([[x['best_original_cos'],x['best_prototype_cos'],x['best_proxy_cos'], max(x['best_original_cos'],x['best_proxy_cos'])] for x in comparisons])
        im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(4), ['Original attributes\nbest cosine', 'Train prototypes\nbest cosine', 'Proxy only\nbest cosine', 'Expanded bank\nbest cosine'])
        ax.set_yticks(range(len(comparisons)), [x['dataset']+' / '+x['name']+(' [shared name]' if x['exact_train_name_overlap'] else '') for x in comparisons], fontsize=8)
        for i in range(len(comparisons)):
            for j in range(4):
                ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=7,color='white' if matrix[i,j]<.6 else 'black')
        fig.colorbar(im, ax=ax, fraction=.025, pad=.03, label='Cosine similarity in full embedding space')
        ax.set_title(f"{r['dataset']} | RemoteCLIP {r['model']}\nShared labels/synonyms remain comparison classes; this is not segmentation accuracy.", pad=15)
        fig.tight_layout()
        fig.savefig(out/(stem+'_cosine.png'),dpi=160)
        fig.savefig(out/(stem+'_cosine.pdf'))
        plt.close(fig)
        for ds in others:
            rows = [x for x in comparisons if x['dataset']==ds]
            stats.append(dict(train=r['dataset'], model=r['model'], comparison_dataset=ds, count=len(rows),
                              mean_original_cos=float(np.mean([x['best_original_cos'] for x in rows])),
                              mean_proxy_cos=float(np.mean([x['best_proxy_cos'] for x in rows])),
                              mean_expanded_gain=float(np.mean([x['expanded_gain'] for x in rows])),
                              improved_count=sum(x['expanded_gain']>1e-6 for x in rows)))
        print(stem, 'plots and CSV saved', flush=True)
    (out/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')

if __name__ == '__main__':
    main()

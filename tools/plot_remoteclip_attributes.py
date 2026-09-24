#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT = Path(__file__).resolve().parents[1] / 'artifacts/remoteclip_raw_attributes_20260910'
results = json.loads((OUT/'projections.json').read_text())
# Same category has the same color in both model panels for each dataset.
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
for method in ['tsne', 'pca']:
    fig = plt.figure(figsize=(18, 13), facecolor='white')
    grid = fig.add_gridspec(2, 3, width_ratios=[1,1,.42], hspace=.32, wspace=.17)
    for row, dataset in enumerate(['DLRSD', 'iSAID']):
        for col, tag in enumerate(['b32','l14']):
            r = next(r for r in results if r['dataset']==dataset and r['tag']==tag)
            ax = fig.add_subplot(grid[row,col])
            coords = np.array(r[method]); labels = np.array(r['labels'])
            colors = plt.get_cmap('tab20').colors
            for i, cls in enumerate(r['classes']):
                mask=labels==i
                ax.scatter(coords[mask,0],coords[mask,1],s=31,color=colors[i],alpha=.85,edgecolors='white',linewidths=.35)
            ax.set_title(f"{dataset} | RemoteCLIP {r['model']}\n{r['count']} original descriptors, {len(r['classes'])} classes", loc='left', fontsize=12, pad=12)
            if method=='pca':
                ax.set_xlabel(f"PC1 ({r['pca_explained_variance'][0]:.1%} variance)")
                ax.set_ylabel(f"PC2 ({r['pca_explained_variance'][1]:.1%} variance)")
            else:
                ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
            ax.grid(alpha=.12); ax.set_axisbelow(True)
        legend_ax=fig.add_subplot(grid[row,2]); legend_ax.axis('off')
        legend_ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=colors[i],markeredgecolor='none',label=cls,markersize=7) for i,cls in enumerate(r['classes'])],loc='center left',frameon=False,title=f'{dataset} categories',labelspacing=.85)
    fig.suptitle(f"RemoteCLIP original attribute embeddings — {method.upper()}", fontsize=20, y=.975)
    fig.text(.06,.025,'Each point = one original attribute sentence; colors = source classes. No clustering or cluster centers.\n'+('Independent t-SNE fits; perplexity=30, seed=42. Axes and distances between panels are not directly comparable.' if method=='tsne' else 'Independent PCA fits on L2-normalized text embeddings; axis labels report explained variance.'),fontsize=11,color='#4b5563')
    fig.subplots_adjust(top=.9,bottom=.10,left=.06,right=.98)
    fig.savefig(OUT/f'raw_attributes_{method}.png',dpi=170)
    fig.savefig(OUT/f'raw_attributes_{method}.pdf')
    plt.close(fig)

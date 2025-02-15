import anndata as ad
import numpy as np
import os, argparse
from scHSC import scHSCModel

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Muraro', help='dataset name')
parser.add_argument('--t', type=int, default=1, help='number of laplacian filtering')
parser.add_argument('--highly_genes', type=int, default=2000, help='number of highly variable genes')
parser.add_argument('--k', type=int, default=18, help='number of nearest neighbors')
parser.add_argument('--target_clusters', type=int, default=0, help='number of clusters to assign')
args = parser.parse_args()
dataset = args.dataset
t = args.t
highly_genes = args.highly_genes
k = args.k
target_clusters = args.target_clusters

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
file = f'./temp/scHSC.csv'
save_dir = './temp'

try:
    adata_raw = ad.read(f'./data/{dataset}.h5ad')
    adata_raw.raw=adata_raw

    assert target_clusters != 0 or 'cluster' in adata_raw.obs, \
    "Either target_clusters must be specified, or 'cluster' must exist in adata_raw.obs."
    nclusters = len(np.unique(adata_raw.obs['cluster'])) if target_clusters == 0 else target_clusters

    log_path = f"{save_dir}/{dataset}_log.txt"

    schsc = scHSCModel(log_path=log_path, info=True)
    adata = schsc.preprocess(adata_raw, t = t, k = k, highly_genes = highly_genes, preprocessed = False, lap__filter = True, approx = True, lap_sparse = True) 
    schsc.train(adata, target_clusters = nclusters)

    from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
    from scHSC.utils import accuracy
    ari = adjusted_rand_score(adata.obs["reassign_cluster"],adata.obs["cluster"])
    nmi = normalized_mutual_info_score(adata.obs["reassign_cluster"],adata.obs["cluster"])
    acc = accuracy(adata.obs["reassign_cluster"],adata.obs["cluster"])
    adata.write(f"{save_dir}/{dataset}_schsc.h5ad", compression = 'gzip')
    with open(file, 'a+') as f:
        f.write(f'{dataset},{nmi},{ari},{acc} \n')
except Exception as e:
    with open(file, 'a+') as f:
        f.write(f'{dataset},{e} \n')

import anndata as ad
import numpy as np
import os, argparse
from scHSC import scHSCModel

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Pollen', help='dataset name')
args = parser.parse_args()
dataset = args.dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
file = f'./temp/scHSC.csv'
save_dir = './temp'

try:
    adata_raw = ad.read(f'./data/{dataset}.h5ad')
    adata_raw.raw=adata_raw
    nclusters = len(np.unique(adata_raw.obs['cluster']))

    log_path = f"{save_dir}/{dataset}_log.txt"

    schsc = scHSCModel(log_path=log_path, info=True)
    adata = schsc.preprocess(adata_raw, preprocessed = False, lap__filter = True, approx = True, lap_sparse = True, t=1) 
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
        f.write(f'{dataset}{e} \n')
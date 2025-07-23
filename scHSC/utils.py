import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import anndata
import gc
import scipy.sparse as sp
from torch.utils.data import Dataset, Sampler
from scipy.optimize import linear_sum_assignment
from .graph_function import get_appro_adj
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def setup_seed(seed):
    """
    setup random seed to fix the result
    :param seed: random seed
    :return:  None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class InfiniteRandomSampler(Sampler):
    def __init__(self, dataset_size, batch_size):
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield from np.random.choice(self.dataset_size, self.batch_size, replace=False)

    def __len__(self):
        return int(1e12)  


class GraphDataset(Dataset):
    def __init__(self, adata):
        self.laplacian_filtered = adata.uns['laplacian_filtered']
        self.adjacency_matrix = adata.uns['adjacency_matrix']
        # self.raw_count = adata.uns['raw_counts']
        # self.raw_count = adata.raw[:, adata.raw.var['highly_variable']].X
        self.raw_count = adata.raw[:, adata.var_names[adata.var['highly_variable']]].X

        self.size_factor = adata.obs['size_factors']

    def __len__(self):
        return self.laplacian_filtered.shape[0]
        


    def __getitem__(self, idx):

        if sp.issparse(self.raw_count):
            raw = self.raw_count[idx, :].toarray().flatten()
        else:
            raw = self.raw_count[idx]

        if sp.issparse(self.adjacency_matrix):
            adj_row = self.adjacency_matrix[idx, :].toarray().flatten()
        else:
            adj_row = self.adjacency_matrix[idx]

        return {
            'laplacian_filtered': torch.tensor(self.laplacian_filtered[idx], dtype=torch.float32),
            # 'adjacency_matrix': torch.tensor(self.adjacency_matrix[idx], dtype=torch.float32),
            'adjacency_matrix': torch.tensor(adj_row, dtype=torch.float32),
            'raw_count': torch.tensor(raw, dtype=torch.int16),
            'size_factor': torch.tensor(self.size_factor[idx], dtype=torch.float32)
        }

def accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    param y: true labels, numpy.array with shape `(n_samples,)`
    param y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def compute_compactness_separation(X, labels, scale = False, normalize = False):
    """
    Compute Compactness and Separation based on X and labels，

    """
    if scale:
        X = StandardScaler().fit_transform(X)

    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    centers = np.zeros((n_clusters, X.shape[1]))

    for i, k in enumerate(unique_labels):
        centers[i] = X[labels == k].mean(axis=0)

    compactness = 0.0
    for i, k in enumerate(unique_labels):
        cluster_points = X[labels == k]
        dists = np.linalg.norm(cluster_points - centers[i], axis=1)
        compactness += np.sum(dists ** 2)

    separation = np.sum(pairwise_distances(centers, metric='euclidean')**2) / 2  

    # ratio = separation / compactness if compactness > 0 else np.inf
    if normalize:
        return compactness/X.shape[0], separation/n_clusters
    
    return compactness, separation

def phi(feature, target_clusters=8, leiden = False, louvain = True, min_resolution = 0, max_resolution = 5, log = None, pre_adj = None): 
    """
    The `phi` function takes in a feature matrix, true labels, and clustering parameters, and returns
    predicted labels and cluster centers.
    
    :param feature: The feature is a numpy array representing the input features for each data point. shape (n_samples, n_features)
    :param target_clusters: target number of clusters 
    :param leiden: A boolean indicating whether to use the Leiden clustering method. 
    :param louvain: A boolean indicating whether to use the Louvain clustering method. 
    :param min_resolution: The minimum resolution parameter for the clustering algorithm. defaults to 0 (optional)
    :param max_resolution: The maximum resolution parameter for the clustering algorithm. defaults to 5 (optional)
    :param plot: The `plot` parameter is a boolean flag that determines whether or not to generate and save plots. 
    :param fig_dir: string that specifies the directory where the figures  will be saved if the `plot` parameter is set to `True`.

    :return: 
    predicted labels and the centers of each cluster. Both values are returnedas numpy arrays.
    """
    assert leiden + louvain == 1, "Specify one clustering method: leiden or louvain."

    feature = feature.cpu().detach().numpy()
    predict_labels = binary_search(feature, target_clusters, leiden = leiden, louvain = louvain, 
                                min_resolution = min_resolution, max_resolution = max_resolution, log = log, pre_adj = pre_adj)
    centers = np.vstack([np.mean(feature[predict_labels == class_val], axis=0) for class_val in np.unique(predict_labels)])
    
    return predict_labels.astype(np.float32), centers.astype(np.float32)
    
def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix

    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not

    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + torch.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)
    return norm_adj

def normalize(adata, copy=True, highly_genes = 2000, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True, info = True, log = None):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
        
    if filter_min_counts:
        if info:
            log.info('Filtering genes and cells...')
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors:
        if info:
            log.info('Normalizing data...')
        sc.pp.normalize_total(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        if info:
            log.info("Log1p data...")
        sc.pp.log1p(adata)

    if highly_genes != None:
        if info:
            log.info("Selecting HVG(n_top_genes={})...".format(highly_genes))
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)

    if normalize_input:
        if info:
            log.info("Scaling data...")
        sc.pp.scale(adata)
    return adata

def laplacian_filtering(A, X, t):
    """
    performs Laplacian filtering on a given matrix A and input vector X for a specified
    number of iterations.
    
    :param A: adjacency matrix. 
    :param X: input data. 
    :param t: number of iterations to perform in the Laplacianfiltering algorithm. 

    :return: the filtered matrix X after applying the Laplacian filtering algorithm for t iterations.
    """
    A_tmp = A - torch.diag_embed(torch.diag(A))
    A_norm = normalize_adj(A_tmp, self_loop=True, symmetry=True) # D^{-0.5} A D^{-0.5}
    # I = torch.eye(A.shape[0])
    # L = I - A_norm
    for _ in range(t):
        #X = (I - L) @ X
        X = A_norm @ X
    return X

# def laplacian_filtering_sparse(adj, X, t):
#     adj = adj + np.eye(adj.shape[0])
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(0))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^{-0.5}
#     adj = torch.tensor(d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).toarray()).float() # D^{-0.5} A D^{-0.5}
#     for _ in range(t):
#         X = adj @ X
#     return X.float()

def laplacian_filtering_sparse(adj, X, t):
    # adj = adj + np.eye(adj.shape[0])
    # adj = sp.coo_matrix(adj)

    if not sparse.issparse(adj):
        adj = sparse.csr_matrix(adj)

    adj = adj + sparse.identity(adj.shape[0], format="csr")
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^{-0.5}
    laplacian = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt # D^{-0.5} A D^{-0.5}
    laplacian = laplacian.astype(np.float32)  # ensure h5py compatibility
    for _ in range(t):
        X = laplacian @ X

    return np.array(X, dtype=np.float32)

from scipy import sparse
from scipy.sparse.linalg import aslinearoperator
# from memory_profiler import profile

# @profile
# def laplacian_filtering_sparse(adj, X, t):
#     # adj = adj + sparse.eye(adj.shape[0])
#     # adj = sp.coo_matrix(adj)

#     if not sparse.issparse(adj):
#         adj = sparse.csr_matrix(adj)

#     # Avoid creating dense identity matrix
#     adj = adj + sparse.identity(adj.shape[0], format="csr")

#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^{-0.5}
#     laplacian = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt # D^{-0.5} A D^{-0.5}

#     laplacian = aslinearoperator(laplacian)

#     # laplacian = laplacian.astype(np.float32)  # ensure h5py compatibility

#     for _ in range(t):
#         X = laplacian @ X
    # return np.array(X, dtype=np.float32)



    
def binary_search(feature, target_clusters, leiden = False, louvain = False, min_resolution = 0, max_resolution = 5, tol=1e-8, log = None, pre_adj = None):
    """
    performs binary search to find the optimal resolution parameter for clustering using either the Leiden or Louvain algorithm.
    
    :param feature:  input feature matrix for clustering. 
    :param true_labels:  array containing the true cluster labels for each data point
    :param leiden: A boolean parameter indicating whether to use the Leiden algorithm for clustering. 
    :param louvain: A boolean parameter indicating whether to use the Louvain algorithm for clustering.
    :param min_resolution: minimum resolution parameter 
    :param max_resolution: maximum resolution parameter 
    :param tol: tolerance value used to determine when to stop the binary search. 
    :param plot: boolean indicating whether or not to generate a plot of the clustering results. 
    :param fig_dir:  string that specifies the directory where the figures will be saved.
    
    :return: predicted labels for the input feature
    """

    adata = anndata.AnnData(feature)
    if adata.n_vars > 256:
        sc.pp.pca(adata)
    
    if adata.X.shape[0] > 15000:
        if log:
            log.info('Constructing approximate adjacency matrix...')
        appro_adj = get_appro_adj(feature)
        from scipy.sparse import csr_matrix
        adata.obsp["connectivities"] = csr_matrix(appro_adj)
        adata.uns["neighbors"] = {"connectivities_key": "connectivities"}
        if adata.X.shape[0] > 100000:
            max_resolution = 0.1
    else:
        if log:
            log.info('Finding neighbors...')
        sc.pp.neighbors(adata)

    # set initial search scope
    low = min_resolution
    high = max_resolution

    function_list = {'leiden':sc.tl.leiden, 'louvain':sc.tl.louvain}
    if leiden:
        clustering_method = 'leiden'
    elif louvain:
        clustering_method = 'louvain'
    else:
        raise RuntimeError('No clustering method specified.')

    function_list[clustering_method](adata, resolution = high)
    # print('initial maximum clusters:', len(np.unique(adata.obs[clustering_method])))
    binary_iter = 0
    while (len(np.unique(adata.obs[clustering_method])) < target_clusters) and (binary_iter < 5):
        high = 2 * high
        # print(f'No suitable resolution in the given range. Try to enlarge it to {high}.')
        function_list[clustering_method](adata, resolution = high)
        binary_iter += 1
    if binary_iter == 5:
        raise RuntimeError('No suitable resolution in the given parameters.')

    while (high - low) > tol:
        mid = (low + high) / 2.0
        temp = [(1,0)]
        # clustering using current mid
        function_list[clustering_method](adata, resolution=mid)
        num_clusters = len(np.unique(adata.obs[clustering_method]))
        if log:
            log.info(f'Trying resolution {mid:.6f} and obtain {num_clusters:<2d} clusters...')
        
        # adjust the search scope according to the clustering results and the number of target
        if num_clusters < target_clusters:
            if num_clusters > temp[-1][0]:
                temp.append((num_clusters,mid))
            low = mid
        elif num_clusters > target_clusters:
            high = mid
        else:
            predict_labels = np.array(adata.obs[clustering_method].astype(int))

            del adata
            gc.collect()
            if log:
                log.info(f'Targeting Resolution {mid:.6f}...')
            return predict_labels
        
    # If fail, use the results of final iteration
    reso = temp[-1][1]
    function_list[clustering_method](adata, resolution=reso)
    # print(f'Stop iteration and reverse to the nums of clusters : {len(np.unique(adata.obs["leiden"]))}')

    predict_labels = np.array(adata.obs[clustering_method].astype(int))

    del adata
    gc.collect()
    print('final reso', reso)
    return predict_labels


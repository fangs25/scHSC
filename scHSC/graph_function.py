import numpy as np
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from annoy import AnnoyIndex
from scipy.sparse import lil_matrix

def get_adj(count, k=15, pca=50, mode="connectivity"):

    if pca and pca < count.shape[1]:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="cosine", include_self=False)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10


def get_appro_adj(count, k=15, pca=50,):
    if pca and pca < count.shape[1]:
        countp = dopca(count, dim=pca)
    else:
        countp = count
        
    tree = AnnoyIndex(countp.shape[1], metric="angular")
    tree.set_seed(5)
    for i in range(countp.shape[0]):
        tree.add_item(i, countp[i, :])
    tree.build(60)
    
    # A = np.zeros((countp.shape[0], countp.shape[0]))
    # for i in range(countp.shape[0]):
    #     indices = tree.get_nns_by_vector(countp[i, :], k, search_k=-1) 
    #     A[i, indices] = 1
    # return A
    
    n = countp.shape[0]
    A = lil_matrix((n, n), dtype=np.float32)
    for i in range(n):
        indices = tree.get_nns_by_vector(countp[i, :], k)
        A[i, indices] = 1

    return A.tocsr()
    
    

from .network import HSCNetwork, ZINBLoss, EarlyStopping
from .graph_function import get_adj,get_appro_adj
from .utils import *
from .logger import create_logger     
from tqdm import tqdm

import torch
from torch import optim
import torch.nn.functional as F
import os
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
import warnings
from torch.utils.data import DataLoader, BatchSampler
warnings.filterwarnings("ignore")
import numpy as np
import scipy
from scipy.sparse import csr_matrix


class scHSCModel:
    def __init__(self, log_path = './results/log.txt', info = True, ):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.info = info

        if not os.path.exists(os.path.dirname(os.path.abspath(log_path))): 
            os.makedirs(os.path.dirname(os.path.abspath(log_path)))

        self.log = create_logger('scHSC',fh=log_path)
        if self.info:
            self.log.info('Create log file...')
            self.log.info('Create scHSCModel Object Done...')

    def preprocess(self, adata, t=1, highly_genes = 2000, k = 18, batch_str = None, lap__filter = True, approx = True, lap_sparse = True,
                   filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
        setup_seed(5)

        if adata.X.shape[0] > 15000:
            if self.info:
                self.log.info('Sparsing raw data...')
            adata.X = scipy.sparse.csr_matrix(adata.X)

        if self.info:
            self.log.info('Preprocessing data...')

        if size_factors or normalize_input or logtrans_input:
            adata.raw = adata.copy()
        else:
            adata.raw = adata
            
        if filter_min_counts:
            if self.info:
                self.log.info(f'Filtering genes({adata.X.shape[1]}) and cells({adata.X.shape[0]})...')
            sc.pp.filter_genes(adata, min_counts=1)
            sc.pp.filter_cells(adata, min_counts=1)

        if size_factors:
            if self.info:
                self.log.info('Normalizing data...')
            sc.pp.normalize_total(adata)
            adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        else:
            adata.obs['size_factors'] = 1.0

        if logtrans_input:
            if self.info:
                self.log.info("Log1p data...")
            sc.pp.log1p(adata)

        if highly_genes != None:
            if self.info:
                self.log.info("Selecting HVG(n_top_genes={})...".format(highly_genes))
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
            # adata.uns['raw_counts'] = adata.raw[:, adata.var_names[adata.var['highly_variable']]].X


        if normalize_input:
            if self.info:
                self.log.info("Scaling data...")
            sc.pp.scale(adata)
        
        if approx:
            if self.info:
                self.log.info("Constructing approximate adjacency matrix using KNN...")
            A = get_appro_adj(adata.X, k = k)
        else:
            if self.info:
                self.log.info("Constructing exact adjacency matrix using KNN...")
            A, _ = get_adj(adata.X, k = k)

        if batch_str != None:
            if self.info:
                self.log.info(f'Constructing adjacency matrix using {A.sum()} KNN pairs...')
            sc.pp.pca(adata) 
            emb_matrix = adata.obsm['X_pca']
            batch_index = adata.obs[batch_str].values
            from .calculate_NN import get_dict_mnn_para
            mnn_inter_batch_approx = get_dict_mnn_para(data_matrix=emb_matrix,batch_index=batch_index,k=10,flag="out",
                                    metric='cosine',approx=True,return_distance=False,njob=1)
            mnn_inter_batch=np.array(mnn_inter_batch_approx)
            NN_matrix = np.zeros((adata.X.shape[0], adata.X.shape[0]))
            indices = tuple(zip(*mnn_inter_batch))
            np.add.at(NN_matrix, indices, 1)
            A = NN_matrix + A
            if self.info:
                self.log.info(f'Correcting adjacency matrix using {NN_matrix.sum()} MNN pairs...')

        adata.uns['adjacency_matrix'] = A

        if lap__filter:
            if lap_sparse:
                if self.info:
                    self.log.info('Laplacian filtering(sparse)...')
                # X_filtered = laplacian_filtering_sparse(A, torch.tensor(adata.X).float(), t)
                X_filtered = laplacian_filtering_sparse(A, adata.X, t)

            else:
                if self.info:
                    self.log.info('Laplacian filtering...')
                X_filtered = laplacian_filtering(A, torch.tensor(adata.X).float(), t)

            adata.uns['laplacian_filtered'] = X_filtered
        else:
            adata.uns['laplacian_filtered'] = adata.X

        return adata



    def train(self, adata, target_clusters = 8, dims = 32, batch_size = 512, drop_rate = 0.5,
              iterations = 100, lr = 0.00001, sep = 3,  #  lr=0.00001!
              alpha = 0.5, beta = 1, tau = 0.9, louvain = True, leiden = False, 
              patience=1, delta=0, warm_up = True, wzinb = 0, init_method = 'kaiming_uniform_'):
        """
        Train the hard sample contrastive learning model.

        Parameters:
        - adata: AnnData object containing the input data.
        - target_clusters: The number of clusters to assign to the data.
        - dims: The dimensionality of the hidden layers in the HSC-Embed model.
        - batch_size: The size of the batches used during training.
        - drop_rate: The dropout rate applied during training.
        - iterations: The number of training iterations.
        - lr: The learning rate for the optimizer.
        - sep: The interval at which to log the training loss.
        - alpha: The weighting factor for the attribute embedding in the comprehensive similarity calculation.
        - beta: The parameter controlling the shape of the pseudo matrix in the pseudo_matrix function.
        - tau: The threshold value for selecting high-confidence samples.
        - louvain: A boolean indicating whether to use the Louvain algorithm for clustering.
        - leiden: A boolean indicating whether to use the Leiden algorithm for clustering.
        - patience: The patience parameter for early stopping.
        - delta: The delta parameter for early stopping.

        Returns:
        - None. The function modifies the `adata` object in-place by adding new obsm fields and obs fields.
        """
        self.batch_size = int(np.min((adata.X.shape[0] - 1, batch_size)))
        self.dims = dims
        if self.info:
            self.log.info(f"Initializing Dataset...")
        dataset = GraphDataset(adata)
        sampler = InfiniteRandomSampler(adata.n_obs, self.batch_size)
        batch_sampler = BatchSampler(sampler, batch_size=self.batch_size, drop_last=True)
        if self.info:
            self.log.info(f"Initializing Train DataLoader...")
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=2,
            pin_memory=True
        )
        if self.info:
            self.log.info(f"Initializing Evaluation DataLoader...")
        eval_dataloader = DataLoader(dataset, batch_size = 5102, shuffle=False, num_workers=2, pin_memory=True)

        if self.info:
            self.log.info(f"Initializing HSCNetwork...")
        self.model = HSCNetwork(input_dim = adata.X.shape[1], dataset_size = adata.X.shape[0], 
                                hidden_dim = self.dims, batch_size = self.batch_size, drop_rate = drop_rate, device = self.device)

        if self.info:
            self.log.info(f"Classify the data into {target_clusters} distinct clusters...")
        self.model.to(self.device)
        init_weights(self.model, method=init_method)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler_ExponentialLR = ExponentialLR(optimizer, gamma=0.98)


        mask_11 = (torch.ones([self.batch_size, self.batch_size]) - torch.eye(self.batch_size)).to(self.device)
        mask_12 = torch.ones([self.batch_size, self.batch_size]).to(self.device)
        
        early_stopping = EarlyStopping(patience=patience, delta=delta)
        self.loss_list = []
        iterator = iter(dataloader)

        for iteration in range(iterations):
            self.model.train()

            batch = next(iterator)

            x_batch = batch['laplacian_filtered'].to(self.device)
            a_batch = batch['adjacency_matrix'].to(self.device)
            raw_batch = batch['raw_count'].to(self.device)
            sf_batch = batch['size_factor'].to(self.device)

            Z1, Z2, E1, E2, _mean, _disp, _pi = self.model(x_batch, a_batch)


            if warm_up:
                Z = (Z1 + Z2) / 2
                P, center = phi(Z, target_clusters, leiden = leiden, louvain = louvain)
                compactness_scale, separation_scale = compute_compactness_separation(Z.detach().cpu().numpy(), P.astype(int).astype(str), scale = True, normalize = True)
                self.log.info(f'iteration:{iteration:<3d}, alpha:{alpha}, lr:{lr}, compactness_norm:{compactness_scale:.6f}, separation_norm:{separation_scale:.6f}')
                
                if compactness_scale > 20 and separation_scale < 100:
                    alpha = 0.95
                    weight = 10
                elif compactness_scale < 20 and separation_scale < 100:
                    weight = 100
                else:
                    alpha = 0.5
                    weight = 0.1

            # self.log.info(f'iteration:{iteration:<3d}, alpha:{alpha}, weight:{weight}')

            ZE_11,ZE_12,ZE_22 = self.comprehensive_similarity(Z1, Z2, E1, E2, alpha)



            contrastive_loss = self.hard_sample_infoNCE(ZE_11, ZE_12, ZE_22, 
                                                   self.model.pos_neg_weight_11, self.model.pos_neg_weight_12, 
                                                   self.model.pos_neg_weight_22, self.model.pos_weight, self.batch_size,
                                                   mask_11, mask_12)
            
            ZINB_Loss = ZINBLoss(pi = _pi, disp = _disp, scale_factor = sf_batch)
            zinb_loss = ZINB_Loss(x = raw_batch, mean = _mean)

            if wzinb > 0:
                Wzinb = torch.tensor(wzinb).to(self.device)
            else:
                Wzinb = (contrastive_loss / zinb_loss).detach().to(self.device) # detach the Wzinb for preventing breaking graph
            loss = (contrastive_loss + Wzinb*zinb_loss)
            total_loss = weight * loss
            total_loss.backward()
            optimizer.step()


            self.loss_list += [loss.item()]
            
            if iteration % sep == 0:
                scheduler_ExponentialLR.step()

                # evaluation mode
                self.model.eval()
                Z1, Z2, E1, E2, _, _, _ = self.model(x_batch, a_batch)

                ZE_11, ZE_12, ZE_22 = self.comprehensive_similarity(Z1, Z2, E1, E2, alpha)

                # fusion and testing
                Z = (Z1 + Z2) / 2

                P, center = phi(Z, target_clusters, leiden = leiden, louvain = louvain)                
                P, center = map(lambda x: torch.tensor(x).to(self.device), (P, center))

                # select high confidence samples
                H, H_mat = self.high_confidence(Z, center, tau = tau) # 1*2M  ((2M*1),(1*2M))  M = tau * N

                M_1, M_mat_11, M_mat_12, M_mat_22 = self.pseudo_matrix(P, ZE_11, ZE_12, ZE_22, beta = beta)

                # update weight
                self.model.pos_weight[H] = M_1[H].data # update M`s confident weight of the true positive sample pairs
                self.model.pos_neg_weight_11[H_mat] = M_mat_11[H_mat].data# update M*M's confident weight among all samples
                self.model.pos_neg_weight_12[H_mat] = M_mat_12[H_mat].data
                self.model.pos_neg_weight_22[H_mat] = M_mat_22[H_mat].data

                self.model.eval()
                if self.info:
                    self.log.info(f'iteration:{iteration:<3d}, loss:{loss:.6f}')
                
                # if self.info:
                    # self.log.info(f'iteration:{iteration:<3d}, loss:{loss:.6f}, alpha:{alpha}, lr:{lr}, nmi:{nmi:.6f}, ari:{ari:.6f}, acc:{acc:.6f}, ss:{ss:.6f}, chs:{chs:.6f}, compactness:{compactness:.6f}, compactness_scale:{compactness_scale:.6f}, compactness_norm:{compactness_norm:.6f}, separation:{separation:.6f}, separation_scale:{separation_scale:.6f}, separation_norm:{separation_norm:.6f}')
                    # self.log.info(f'iteration:{iteration:<3d}, loss:{loss:.6f}, nmi:{nmi:.6f}, ari:{ari:.6f}, acc:{acc:.6f}, compactness_norm:{compactness_norm:.6f}, separation_norm:{separation_norm:.6f}')

                early_stopping(loss)

                if early_stopping.early_stop :
                    self.log.info(f"----------Early stopping in {iteration:<3d}th/{iterations:<3d}.--------")
                    self.log.info("Training complete!")

                    self.model.eval()
                    Z_full = self.predict_full(eval_dataloader)
                    
                    clusters, _ = phi(Z_full, target_clusters, leiden = leiden, louvain = louvain, log = self.log)
                    self.log.info(f"----------Clustering Done!--------")

                    break
                

            if iteration == iterations - 1: # Final Evaluation
                self.model.eval()
                Z_full = self.predict_full(eval_dataloader)

                clusters, _ = phi(Z_full, target_clusters, leiden = leiden, louvain = louvain,log=self.log)


        adata.obsm["X_schsc"] = Z_full.detach().cpu().numpy()
        adata.obs["reassign_cluster"] = clusters.astype(int).astype(str)
        adata.obs["reassign_cluster"] = adata.obs["reassign_cluster"].astype("category")
    
    
    def predict_full(self, eval_dataloader):
        self.model.eval()

        total_size = len(eval_dataloader.dataset)
        device = self.device
        Z_full = torch.empty((total_size, self.dims), dtype=torch.float32, device=device)

        with torch.no_grad():
            start = 0
            for batch in tqdm(eval_dataloader, desc="Predicting", leave=True):
                x_batch = batch['laplacian_filtered'].to(device, non_blocking=True)
                Z1, Z2 = self.model.forward_full(x_batch)
                Z = (Z1 + Z2) / 2

                end = start + Z.size(0)
                Z_full[start:end] = Z  
                start = end

        return Z_full.cpu()

    
    
    def comprehensive_similarity(self, Z1, Z2, E1, E2, alpha):
        """
        Calculate the similarity matrices between two sets of vectors Z1 and Z2, 
        and between two sets of vectors E1 and E2, using a weighted combination of matrix products.
        
        :param Z1: The first version of attribute embedding.
        :param Z2: The second version of attribute embedding.
        :param E1: The first version of structure embedding.
        :param E2: The second version of structure embedding.
        :param alpha: Weighting factor that determines the balance between the attribute embedding and structure embedding.

        :return: three similarity matrices: ZE_11, ZE_12, and ZE_22, each with n*n dimension.
        """

        ZE_11 = alpha * Z1 @ Z1.T + (1 - alpha) * E1 @ E1.T
        ZE_12 = alpha * Z1 @ Z2.T + (1 - alpha) * E1 @ E2.T
        ZE_22 = alpha * Z2 @ Z2.T + (1 - alpha) * E2 @ E2.T 
        return ZE_11,ZE_12,ZE_22


    def hard_sample_infoNCE(self, ZE_11, ZE_12, ZE_22, 
                            pos_neg_weight_11, pos_neg_weight_12, pos_neg_weight_22, pos_weight, 
                            node_num, mask_11, mask_12):
        """
        Calculate the information noise-contrastive estimation (infoNCE) loss for a given set of input tensors.
        
        :param ZE_11: The pairwise similarity scores within the first fusion embedding.
        :param ZE_12: The pairwise similarity scores between the first version of fusion embedding and the second.
        :param ZE_22: The pairwise similarity scores within the second fusion embedding.
        :param pos_neg_weight_11: Weight applied to the positive-negative term in the calculation of `pos_neg_11`.
        :param pos_neg_weight_12: Weight applied to the positive-negative term in the calculation of `pos_neg_12`. 
        :param pos_neg_weight_22: Weight applied to the positive-negative term in the calculation of `pos_neg_22`.
        :param pos_weight: Scalar value that is used to weight the positive-positive term in the calculation of the `pos` variable. 
        :param node_num: Number of nodes in the graph.
        :param mask_11: Binary mask that is applied element-wise to the tensor `ZE_11` and `ZE_22`. 
        :param mask_12: Binary mask that is applied element-wise to the tensor `ZE_12`. 

        :return: the value of `infoNEC`, which is the calculated information Noise-Contrastive Estimation (infoNEC) score.
        """
        pos = torch.exp(torch.diag(ZE_12) * pos_weight)
        
        pos = torch.cat([pos, pos], dim = 0)

        pos_neg_11 = mask_11 * torch.exp(ZE_11 * pos_neg_weight_11)
        pos_neg_12 = mask_12 * torch.exp(ZE_12 * pos_neg_weight_12)
        pos_neg_22 = mask_11 * torch.exp(ZE_22 * pos_neg_weight_22)

        neg = torch.cat(   [torch.sum(pos_neg_11, dim=1) + torch.sum(pos_neg_12, dim=1) - pos[:node_num],
                            torch.sum(pos_neg_22, dim=1) + torch.sum(pos_neg_12, dim=0) - pos[node_num:]], dim = 0)
        infoNEC = (-torch.log(pos / (pos + neg))).sum() / (2 * node_num)
        return infoNEC


    def square_euclid_distance(self, Z, center):
        """
        Calculate the squared Euclidean distance between a set of points Z and a center point.
        
        :param Z: Z is a matrix representing a set of points in a Euclidean space. Each row of Z represents a point.
        :param center: Numpy array representing the center points with shape `(k, d)`, 
        where `k` is the number of center points and `d` is the dimensionality of each center point.

        :return: the squared Euclidean distance between the input matrix Z and the center matrix.
        """

        ZZ = (Z * Z).sum(-1).reshape(-1, 1).repeat(1, center.shape[0])
        CC = (center * center).sum(-1).reshape(1, -1).repeat(Z.shape[0], 1)
        ZZ_CC = ZZ + CC
        ZC = Z @ center.T
        distance = ZZ_CC - 2 * ZC
        return distance 


    def high_confidence(self, Z, center, tau):
        """
        Calculate the high confidence samples based on their distance from a center point.
        
        :param Z: a set of data embedding. 
        :param center: Numpy array representing the center points with shape `(k, d)`, 
        where `k` is the number of center points and `d` is the dimensionality of each center point.
        :param tau: Threshold value that determines the confidence level. 
        It is used to select the top-k distances based on the normalized distance values. 
        The top-k distances are then used to identify the high-confidence samples. Defaults to 0.9 (optional).

        :return: two values: H and H_mat. H is a tensor containing the indices of the high-confidence
        samples, while H_mat is a numpy array representing the submatrix of Z corresponding to the
        high-confidence samples.
        """

        distance_norm = torch.min(F.softmax(self.square_euclid_distance(Z, center), dim=1), dim=1).values
        value, _ = torch.topk(distance_norm, int(Z.shape[0] * (1 - tau)))
        index = torch.where(distance_norm <= value[-1],
                                    torch.ones_like(distance_norm), torch.zeros_like(distance_norm)) 

        H = torch.nonzero(index).reshape(-1, ) # 1*M
        H_mat = np.ix_(H.cpu(), H.cpu())
        return H, H_mat 


    def pseudo_matrix(self, P, ZE_11, ZE_12, ZE_22, beta):
        """
        The function `pseudo_matrix` takes in several input tensors and computes various normalized matrices
        based on them.
        
        :param P: A tensor of shape (N, N) representing a matrix
        :param ZE_11: ZE_11 is a tensor representing the first element of the ZE matrix
        :param ZE_12: ZE_12 is a tensor representing the element ZE_12 in a matrix
        :param ZE_22: ZE_22 is a tensor representing the values of ZE (Zero Energy) for a specific condition
        or scenario
        :param beta: The parameter `beta` is a scalar value that controls the shape of the pseudo matrix. It
        determines the degree of non-linearity in the transformation of the input matrix `P` to the output
        matrix `M_mat_12`. A higher value of `beta` will result in a more non-linear, defaults to 1
        (optional)
        :return: four variables: M_1, M_mat_11, M_mat_12, and M_mat_22.
        """

        Q = (P == P.unsqueeze(1)).float().to(self.device)
        ZE_max = torch.cat([ZE_11, ZE_12, ZE_22]).max()
        ZE_min = torch.cat([ZE_11, ZE_12, ZE_22]).min()

        ZE_11_norm = (ZE_11 - ZE_min) / (ZE_max - ZE_min)
        ZE_12_norm = (ZE_12 - ZE_min) / (ZE_max - ZE_min)
        ZE_22_norm = (ZE_22 - ZE_min) / (ZE_max - ZE_min)

        M_mat_11 = torch.abs(Q - ZE_11_norm) ** beta
        M_mat_12 = torch.abs(Q - ZE_12_norm) ** beta
        M_mat_22 = torch.abs(Q - ZE_22_norm) ** beta

        M_1 = torch.diag(M_mat_12)
        return M_1, M_mat_11, M_mat_12, M_mat_22
    
def init_weights(model, method='kaiming_uniform'):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

            if m.bias is not None:
                nn.init.zeros_(m.bias)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HSCNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, dataset_size, drop_rate = 0.5, device = 'cuda:0'):
        super(HSCNetwork, self).__init__()
        # 2000 - 256 - 32
        self.AE1 = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop_rate), nn.Linear(256, hidden_dim))
        self.AE2 = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop_rate), nn.Linear(256, hidden_dim))
        
        # 500 - 256 - 32
        self.SE1 = nn.Sequential(nn.Linear(dataset_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop_rate), nn.Linear(256, hidden_dim))
        self.SE2 = nn.Sequential(nn.Linear(dataset_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop_rate), nn.Linear(256, hidden_dim))

        # 32 - 2000
        self.dec_mean = nn.Sequential(nn.Linear(hidden_dim, input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(hidden_dim, input_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

        self.pos_weight = torch.ones(batch_size).to(device) # N*1
        self.pos_neg_weight_11 = torch.ones([batch_size, batch_size]).to(device) # N*N
        self.pos_neg_weight_12 = torch.ones([batch_size, batch_size]).to(device)
        self.pos_neg_weight_22 = torch.ones([batch_size, batch_size]).to(device)

        self.activate = lambda x: x

    def forward(self, x, A):
        Z1 = F.normalize(self.activate(self.AE1(x)), dim=1, p=2) # attribute encoder
        Z2 = F.normalize(self.activate(self.AE2(x)), dim=1, p=2) 

        E1 = F.normalize(self.SE1(A), dim=1, p=2) # structure encoder
        E2 = F.normalize(self.SE2(A), dim=1, p=2)

        _mean = self.dec_mean((Z1+Z2)/2) # ZINB decoder
        _disp = self.dec_disp((Z1+Z2)/2)
        _pi = self.dec_pi((Z1+Z2)/2)

        return Z1, Z2, E1, E2, _mean, _disp, _pi
    
    def forward_full(self, x):  
        Z1_full = F.normalize(self.activate(self.AE1(x)), dim=1, p=2) # full data predict
        Z2_full = F.normalize(self.activate(self.AE2(x)), dim=1, p=2) 

        return Z1_full, Z2_full


class ZINBLoss(nn.Module):
    def __init__(self, pi, disp, scale_factor=1.0, ridge_lambda=0.0):
        super(ZINBLoss, self).__init__()
        self.pi = pi
        self.disp = disp
        self.scale_factor = scale_factor
        self.ridge_lambda = ridge_lambda

    def forward(self, x, mean):
        eps = 1e-10
        scale_factor = self.scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(self.disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+self.disp+eps)
        t2 = (self.disp+x) * torch.log(1.0 + (mean/(self.disp+eps))) + (x * (torch.log(self.disp+eps) - torch.log(mean+eps)))
        nb_final = _nan2inf(t1 + t2)
        nb_case = nb_final - torch.log(1.0-self.pi+eps)

        zero_nb = torch.pow(self.disp/(self.disp+mean+eps), self.disp)
        zero_case = -torch.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)

        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        if self.ridge_lambda > 0:
            ridge = self.ridge_lambda*torch.square(self.pi)
            result += ridge
            
        result = torch.mean(result)
        return result
    

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta
        self.early_stop = False

    def __call__(self, loss):
        score = -loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0    

import numpy as np
from .graph import update_sv, L2_distance_1
from .initialization import init_H
from .utils import pos, neg

def uimufslr(X_missing, XF, M, zero_indices, class_num, maxiter=50, neighbor_num=5, 
             alpha=10, beta=1, tau=1, lambda_=1, gamma=10000, random_state=None):
    """Main UIMUFSLR Algorithm."""
    view_num = len(X_missing)
    sample_num = XF.shape[1]
    
    X, dim_num, instance_num = [], [], []
    for v in range(view_num):
        dv, nv = X_missing[v].shape
        dim_num.append(dv)
        instance_num.append(nv)
        X.append(X_missing[v] @ M[v])
        
    S = []
    for v in range(view_num):
        S_v = update_sv(X_missing[v], class_num, neighbor_num, islocal=1, random_state=random_state)
        S.append(M[v].T @ S_v @ M[v])
        
    SC = np.zeros((sample_num, sample_num))
    B, C, D = [], [], []
    
    for v in range(view_num):
        SC += S[v]
        bb = np.ones(sample_num)
        bb[zero_indices[v]] = instance_num[v] / sample_num
        B.append(np.diag(bb))
        C.append(np.ones((dim_num[v], class_num)))
        D.append(np.eye(dim_num[v]))
        
    YT = init_H(XF.T, class_num, method='k_means', random_state=random_state)
    Y = YT.T 
    
    for it in range(maxiter):
        SBA = np.zeros((sample_num, sample_num))
        BA = np.zeros((sample_num, sample_num))
        A = []
        for v in range(view_num):
            tem = SC - S[v]
            col_sum_sq = np.sum(tem**2, axis=0)
            A_v = np.diag(0.5 / np.sqrt(col_sum_sq + np.finfo(float).eps))
            A.append(A_v)
            SBA += S[v] @ B[v] @ A_v
            BA += B[v] @ A_v
            
        YY = L2_distance_1(Y, Y)
        num_sc = pos(2 * lambda_ * SBA) + neg(2 * lambda_ * SC @ BA + tau * YY)
        den_sc = np.maximum(pos(2 * lambda_ * SC @ BA + tau * YY) + neg(2 * lambda_ * SBA), np.finfo(float).eps)
        SC = SC * num_sc / den_sc
        SC = SC @ np.diag(1.0 / (np.sum(SC, axis=0) + 1e-16))
        
        SC_sym = (SC + SC.T) / 2.0
        L = np.diag(np.sum(SC_sym, axis=0)) - SC_sym
        
        for v in range(view_num):
            XXT = X[v] @ X[v].T
            Cu = X[v] @ Y.T + beta * XXT @ C[v]
            ones_c = np.ones((class_num, class_num))
            Cd = XXT @ C[v] + alpha * D[v] @ C[v] + beta * XXT @ C[v] @ ones_c
            
            C[v] = C[v] * (pos(Cu) + neg(Cd)) / np.maximum(pos(Cd) + neg(Cu), 1e-16)
            tempD = 0.5 / np.sqrt(np.sum(C[v]**2, axis=1, keepdims=True) + 1e-16)
            D[v] = np.diag(tempD.flatten())
            
        CTX = sum(C[v].T @ X[v] for v in range(view_num))
        Yu = CTX + 2 * gamma * Y
        Yd = Y + 2 * tau * Y @ L + 2 * gamma * Y @ Y.T @ Y
        
        Y = Y * (pos(Yu) + neg(Yd)) / np.maximum(pos(Yd) + neg(Yu), 1e-16)
        Y = Y @ np.diag(np.sqrt(1.0 / (np.diag(Y.T @ Y) + 1e-16)))
        
    WW = np.vstack(C)
    scores = np.sum(WW**2, axis=1)
    ranking = np.argsort(scores)[::-1]
    
    return ranking, C, Y
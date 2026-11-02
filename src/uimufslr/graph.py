import numpy as np
from scipy.linalg import eigh
from .utils import L2_distance_1, proj_simplex

def update_sv(X: np.ndarray, c: int, k: int, islocal: int = 1, 
              max_iter: int = 30, random_state: int = None) -> np.ndarray:
    """
    Graph learning for multiview clustering.
    Equivalent to MATLAB's Updata_Sv.m
    X: (d, num) where columns are samples.
    """
    num = X.shape[1]
    distX = L2_distance_1(X, X)
    
    idx = np.argsort(distX, axis=1)
    distX1 = np.take_along_axis(distX, idx, axis=1)
    
    A = np.zeros((num, num))
    for i in range(num):
        di = distX1[i, 1:k+2]
        id_ = idx[i, 1:k+2]
        denom = k * di[k] - np.sum(di[:k]) + 1e-10
        A[i, id_] = (di[k] - di) / denom
        
    A0 = (A + A.T) / 2.0
    D0 = np.diag(np.sum(A0, axis=1))
    L0 = D0 - A0
    
    # eigh returns eigenvalues in ascending order (smallest first)
    eigvals, eigvecs = eigh(L0)
    F = eigvecs[:, :c] 
    
    lambda_ = 1.0
    for it in range(max_iter):
        distf = L2_distance_1(F.T, F.T)
        
        A_new = np.zeros((num, num))
        for i in range(num):
            if islocal == 1:
                idxa0 = idx[i, 1:k+1]
            else:
                idxa0 = np.arange(num)
                
            dfi = distf[i, idxa0]
            ad = -dfi / (2 * lambda_)
            A_new[i, idxa0] = proj_simplex(ad, k=1)
            
        A_new = (A_new + A_new.T) / 2.0
        D = np.diag(np.sum(A_new, axis=1))
        L = D - A_new
        
        F_old = F
        eigvals, eigvecs = eigh(L)
        F = eigvecs[:, :c]
        
        fn1 = np.sum(eigvals[:c])
        fn2 = np.sum(eigvals[:c+1]) if c+1 <= num else np.sum(eigvals)
        
        if fn1 > 1e-11:
            lambda_ /= 2.0
        elif fn2 < 1e-11:
            lambda_ *= 2.0
            F = F_old
        else:
            break
            
    A_new = (A_new + A_new.T) / 2.0
    return A_new
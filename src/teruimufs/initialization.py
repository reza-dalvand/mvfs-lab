import numpy as np
from .graph import constructW, optSigma
from .clustering import litekmeans

def ADA_ORTH_init(X, c):
    t = optSigma(X)**2
    A = constructW(X, k=5, WeightMode='HeatKernel', t=t)
    diag_ele_arr = np.sum(A, axis=1)
    diag_ele_arr_t = diag_ele_arr**(-0.5)
    L = np.eye(X.shape[0]) - np.diag(diag_ele_arr_t) @ A @ np.diag(diag_ele_arr_t)
    L = (L + L.T) / 2
    eigval, eigvec = np.linalg.eigh(L)
    idx = np.argsort(eigval)
    eigvec = eigvec[:, idx[:c]]
    eigvec = eigvec / np.sqrt(np.sum(eigvec**2, axis=1, keepdims=True) + 1e-12)
    labels = litekmeans(eigvec, c, Replicates=10)
    H = np.random.rand(X.shape[0], c)
    for i in range(X.shape[0]):
        H[i, :] = 0
        H[i, labels[i]] = 1
    H = H + 0.2
    return A, H

def init_H(XX, c, method='k_means'):
    n = XX.shape[0]
    if method == 'SC':
        _, H = ADA_ORTH_init(XX, c)
    elif method == 'k_means':
        labels = litekmeans(XX, c, MaxIter=20, Replicates=2)
        H = np.zeros((n, c))
        for i in range(n): H[i, labels[i]] = 1
        H = H / np.sqrt(np.sum(H, axis=1, keepdims=True))
    return H
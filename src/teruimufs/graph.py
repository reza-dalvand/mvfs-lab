import numpy as np
from .utils import EuDist2, L2_distance_1

def optSigma(X):
    dist = EuDist2(X, X)
    return np.median(dist.flatten())

def constructW(fea, k=5, WeightMode='HeatKernel', t=1.0):
    nSmp = fea.shape[0]
    D = EuDist2(fea, fea)
    idx = np.argsort(D, axis=1)
    dump = np.sort(D, axis=1)
    
    dump = dump[:, 1:k+1]
    idx = idx[:, 1:k+1]
    
    if WeightMode == 'HeatKernel':
        dump = np.exp(-dump / (2 * t**2))
        
    W = np.zeros((nSmp, nSmp))
    for i in range(nSmp):
        W[i, idx[i, :]] = dump[i, :]
        
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0)
    return W

def EProjSimplex_new(v, k=1):
    v0 = v - np.mean(v) + k / len(v)
    vmin = np.min(v0)
    if vmin < 0:
        lambda_m = 0.0
        v1 = v0
        for _ in range(1000):
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            if npos == 0: break
            g = -npos
            f = np.sum(v1[posidx]) - k
            if abs(f) < 1e-10: break
            lambda_m = lambda_m - f / g
        return np.maximum(v1, 0)
    return v0

def Update_S(A_list, Y_list, V, beta, gamma, delta):
    n = A_list[0].shape[0]
    G_list = [L2_distance_1(Y, Y) for Y in Y_list]
    S = np.zeros((n, n))
    for i in range(n):
        gi = np.zeros(n)
        deltaAi = np.zeros(n)
        for i1 in range(V):
            gi += G_list[i1][i, :]
            deltaAi += 2 * delta[i1] * A_list[i1][i, :]
        gi_temp = gamma / (2 * beta) * gi
        r = (deltaAi - gi_temp) / (2 * V)
        S[i, :] = EProjSimplex_new(r, 1)
    return S

def solve_delta(S, A_list):
    View = len(A_list)
    p = np.zeros(View)
    q = np.zeros(View)
    for v in range(View):
        p[v] = np.trace(A_list[v] @ S.T)
        q[v] = np.trace(A_list[v] @ A_list[v].T)
        
    sum_p_q = np.sum(p / q)
    sum_1_q = np.sum(1 / q)
    g = np.zeros(View)
    for v in range(View):
        g[v] = p[v] / q[v] + (1 - sum_p_q) / (q[v] * sum_1_q)
        
    gmin = np.min(g)
    if gmin < 0:
        miu = 0.0
        sumq = sum_1_q
        der_temp = (1 / sumq) / q
        for _ in range(1000):
            max0 = (miu / sumq) / q - g
            posidx = max0 > 0
            if not np.any(posidx): break
            der = np.sum(der_temp[posidx]) - 1
            f = np.sum(max0[posidx]) - miu
            if abs(f) < 1e-10: break
            miu = miu - f / der
        delta_temp = g - (miu / sumq) / q
        return np.maximum(delta_temp, 0)
    return np.maximum(g, 0)
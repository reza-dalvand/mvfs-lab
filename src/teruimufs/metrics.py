import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

def bestMap(L1, L2):
    nClass = int(max(np.max(L1), np.max(L2)))
    G = np.zeros((nClass, nClass))
    for i in range(nClass):
        for j in range(nClass):
            G[i, j] = np.sum((L1 == i+1) & (L2 == j+1))
    row_ind, col_ind = linear_sum_assignment(-G) # الگوریتم مجارستانی
    newL2 = np.zeros_like(L2)
    for i in range(nClass):
        newL2[L2 == i+1] = col_ind[i] + 1
    return newL2

def MutualInfo(L1, L2):
    nClass = int(max(np.max(L1), np.max(L2)))
    G = np.zeros((nClass, nClass))
    for i in range(nClass):
        for j in range(nClass):
            G[i, j] = np.sum((L1 == i+1) & (L2 == j+1)) + 1e-10
    sumG = np.sum(G)
    P1 = np.sum(G, axis=1) / sumG
    P2 = np.sum(G, axis=0) / sumG
    H1 = np.sum(-P1 * np.log2(P1 + 1e-12))
    H2 = np.sum(-P2 * np.log2(P2 + 1e-12))
    P12 = G / sumG
    PPP = P12 / (np.outer(P1, P2) + 1e-12)
    PPP[np.abs(PPP) < 1e-12] = 1
    MI = np.sum(P12 * np.log2(PPP))
    MIhat = MI / max(H1, H2) # دقیقاً مطابق فرمول متلب
    return np.real(MIhat)

def compute_f(T, H):
    N = len(T)
    numT = numH = numI = 0
    for n in range(N-1):
        Tn = (T[n+1:] == T[n])
        Hn = (H[n+1:] == H[n])
        numT += np.sum(Tn)
        numH += np.sum(Hn)
        numI += np.sum(Tn & Hn)
    p = numI / numH if numH > 0 else 1
    r = numI / numT if numT > 0 else 1
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return f, p, r

def EvaluationMetrics(Y, predY):
    Y = np.asarray(Y).flatten()
    predY = np.asarray(predY).flatten()
    uY = np.unique(Y)
    Y0 = np.zeros_like(Y)
    for i, val in enumerate(uY): Y0[Y == val] = i + 1
    Y = Y0
    uY = np.unique(predY)
    predY0 = np.zeros_like(predY)
    for i, val in enumerate(uY): predY0[predY == val] = i + 1
    predY = predY0
    
    pred_classnum = len(np.unique(predY))
    correnum = 0
    for ci in range(1, pred_classnum + 1):
        incluster = Y[predY == ci]
        if len(incluster) > 0:
            counts = np.bincount(incluster)
            correnum += np.max(counts)
    Purity = correnum / len(predY)
    
    res = bestMap(Y, predY)
    ACC = np.sum(Y == res) / len(Y)
    MIhat = MutualInfo(Y, res)
    ARI = adjusted_rand_score(Y, predY)
    Fscore, Precision, Recall = compute_f(Y, predY)
    
    return np.array([ACC, MIhat, Purity, Fscore, Precision, Recall, ARI])
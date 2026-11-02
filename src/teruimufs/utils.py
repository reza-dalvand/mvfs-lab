import numpy as np

def to_numpy_array(x):
    """تبدیل ماتریس‌های sparse یا np.matrix به آرایه استاندارد numpy"""
    if hasattr(x, 'toarray'):
        x = x.toarray()
    if isinstance(x, np.matrix):
        x = np.asarray(x)
    return x

def EuDist2(fea_a, fea_b=None, bSqrt=False):
    fea_a = to_numpy_array(fea_a)
    
    if fea_b is None:
        aa = np.sum(fea_a * fea_a, axis=1, keepdims=True)
        ab = fea_a @ fea_a.T
        D = aa + aa.T - 2 * ab
        D = np.maximum(D, 0)
        if bSqrt: D = np.sqrt(D)
        D = np.maximum(D, D.T) # تضمین تقارن
    else:
        fea_b = to_numpy_array(fea_b)
        aa = np.sum(fea_a * fea_a, axis=1, keepdims=True)
        bb = np.sum(fea_b * fea_b, axis=1, keepdims=True)
        ab = fea_a @ fea_b.T
        D = aa + bb.T - 2 * ab
        D = np.maximum(D, 0)
        if bSqrt: D = np.sqrt(D)
    return D

def L2_distance_1(a, b):
    a = to_numpy_array(a)
    b = to_numpy_array(b)
    
    if a.ndim == 1: a = a.reshape(1, -1)
    if b.ndim == 1: b = b.reshape(1, -1)
    aa = np.sum(a * a, axis=0, keepdims=True)
    bb = np.sum(b * b, axis=0, keepdims=True)
    ab = a.T @ b
    d = aa.T + bb - 2 * ab
    return np.maximum(np.real(d), 0)

def NormalizeFea(fea, row=1):
    fea = to_numpy_array(fea)
    if row:
        feaNorm = np.maximum(1e-14, np.sum(fea**2, axis=1, keepdims=True))
        fea = fea / np.sqrt(feaNorm)
    else:
        feaNorm = np.maximum(1e-14, np.sum(fea**2, axis=0, keepdims=True))
        fea = fea / np.sqrt(feaNorm)
    return fea
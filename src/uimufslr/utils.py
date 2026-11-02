import numpy as np

def pos(x: np.ndarray) -> np.ndarray:
    """Element-wise positive part: (|x| + x) / 2"""
    return np.maximum(x, 0)

def neg(x: np.ndarray) -> np.ndarray:
    """Element-wise negative part: (|x| - x) / 2"""
    return np.maximum(-x, 0)

def L2_distance_1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distance between columns of a and b.
    Equivalent to MATLAB's L2_distance_1.m
    a: (d, n), b: (d, m) -> returns (n, m)
    """
    aa = np.sum(a**2, axis=0)
    bb = np.sum(b**2, axis=0)
    ab = a.T @ b
    d = aa[:, None] + bb[None, :] - 2 * ab
    return np.maximum(np.real(d), 0)

def proj_simplex(v: np.ndarray, k: float = 1.0) -> np.ndarray:
    """
    Euclidean projection of vector v onto the simplex sum(x)=k, x>=0.
    Replaces the iterative Newton method in EProjSimplex_new.m with 
    the mathematically equivalent, highly stable O(N log N) sorting algorithm.
    """
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - k
    ind = np.arange(len(v)) + 1
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.zeros_like(v)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    return np.maximum(v - theta, 0)
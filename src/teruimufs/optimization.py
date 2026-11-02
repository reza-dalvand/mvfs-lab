import numpy as np

def solve_l1l2(W, lambda_):
    E = W.copy()
    n = W.shape[1]
    for i in range(n):
        w = W[:, i]
        nw = np.linalg.norm(w)
        if nw > lambda_:
            E[:, i] = (nw - lambda_) * w / max(nw, 1e-12)
        else:
            E[:, i] = 0
    return E
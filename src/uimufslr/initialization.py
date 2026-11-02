import numpy as np
from sklearn.cluster import KMeans

def init_H(XX: np.ndarray, c: int, method: str = 'k_means', random_state: int = None) -> np.ndarray:
    """
    Initialize cluster indicator matrix H.
    Equivalent to MATLAB's init_H.m
    XX: (n, d) where rows are samples.
    Returns H: (n, c) scaled indicator matrix.
    """
    n = XX.shape[0]
    if method.lower() == 'k_means':
        # n_jobs removed in newer scikit-learn versions
        kmeans = KMeans(n_clusters=c, n_init=2, max_iter=20, random_state=random_state)
        labels = kmeans.fit_predict(XX)
        H = np.zeros((n, c))
        for i in range(n):
            H[i, labels[i]] = 1.0
        cluster_sizes = np.sum(H, axis=0)
        H = H / np.sqrt(cluster_sizes + 1e-10)
        return H
    else:
        raise ValueError(f"Unknown init method: {method}")
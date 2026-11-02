import numpy as np

def construct_uimdata(data, per, random_state=None):
    """Generates unbalanced incomplete multiview data."""
    view_num = len(data)
    ratios = {1: [1.0], 2: [0.5, 1.5], 3: [0.5, 1.0, 1.5], 
              4: [0.25, 0.75, 1.25, 1.75], 5: [0.25, 0.75, 1.0, 1.25, 1.75],
              6: [0.25, 0.5, 1.0, 1.0, 1.5, 1.75]}
    ratio = ratios.get(view_num, [1.0]*view_num)
    per_del = 0.01 * per * np.array(ratio)
    
    rng = np.random.default_rng(random_state)
    n = data[0].shape[1] if data[0].ndim > 1 else data[0].shape[0]
    
    X_missing, zero_indices, one_indices = [], [], []
    for v in range(view_num):
        miss_num = int(np.floor(n * per_del[v]))
        zero_idx = rng.choice(n, miss_num, replace=False)
        one_idx = np.setdiff1d(np.arange(n), zero_idx)
        
        X_v = data[v] if data[v].shape[1] == n else data[v].T
        X_v_miss = X_v[:, one_idx] 
        X_missing.append(X_v_miss)
        zero_indices.append(zero_idx)
        one_indices.append(one_idx)
        
    return X_missing, zero_indices, one_indices

def generate_synthetic_dataset(n_samples=300, view_dims=[50, 60, 70], n_clusters=5, missing_per=20, random_state=None):
    """Generates a synthetic dataset for integration testing."""
    rng = np.random.default_rng(random_state)
    n = n_samples
    labels = rng.integers(0, n_clusters, size=n)
    
    X = []
    for dv in view_dims:
        centers = rng.standard_normal((n_clusters, dv)) * 2.0
        X_v = centers[labels] + rng.standard_normal((n, dv)) * 0.5
        X.append(X_v.T) 
        
    X_missing, zero_indices, _ = construct_uimdata(X, missing_per, random_state=random_state)
    
    M, XF = [], np.zeros((0, n))
    for v in range(len(view_dims)):
        M_v = np.eye(n)
        M_v = np.delete(M_v, zero_indices[v], axis=0)
        M.append(M_v)
        XF = np.vstack([XF, X[v]])
        
    return X_missing, XF, M, zero_indices, labels, n_clusters, n
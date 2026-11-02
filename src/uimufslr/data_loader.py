import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import os

def load_dataset(filepath: str):
    """Loads BBCSport or similar .mat multiview datasets."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    mat = sio.loadmat(filepath)

    # 1. Load and process Labels
    labels = np.asarray(mat['labels']).flatten().astype(int)
    labels = labels - np.min(labels)  # Ensure 0-based class labels
    n = len(labels)
    class_num = len(np.unique(labels))

    # Helper to unwrap and densify MATLAB cell arrays (handles Sparse matrices)
    def extract_matrices(cell_array):
        items = np.asarray(cell_array).flatten()
        matrices = []
        for item in items:
            # Unwrap 1x1 object arrays if present
            if isinstance(item, np.ndarray) and item.dtype == object and item.size == 1:
                item = item.item()
            
            # CRITICAL FIX: Convert Scipy Sparse matrices to dense NumPy arrays
            if sp.issparse(item):
                mat_dense = item.toarray()
            else:
                mat_dense = np.asarray(item, dtype=float)
                
            # Squeeze unnecessary dimensions
            mat_dense = np.squeeze(mat_dense)
            matrices.append(mat_dense)
        return matrices

    # 2. Load X (Complete data per view)
    X = extract_matrices(mat['X'])
    view_num = len(X)

    # Ensure X[v] is strictly (dv, n) [features x samples]
    for v in range(view_num):
        if X[v].ndim == 1:
            if X[v].shape[0] == n:
                X[v] = X[v].reshape(1, n)
            else:
                X[v] = X[v].reshape(-1, 1)
        
        # If the matrix is transposed (n x dv), flip it back to (dv x n)
        if X[v].shape[0] == n and X[v].shape[1] != n:
            X[v] = X[v].T

    # 3. Load zero_indices (Missing sample indices)
    zero_indices = []
    zi_cell = np.asarray(mat['zero_indices']).flatten()
    for zi in zi_cell:
        if isinstance(zi, np.ndarray) and zi.dtype == object and zi.size == 1:
            zi = zi.item()
        zi_arr = np.asarray(zi).flatten().astype(int)
        # MATLAB uses 1-based indexing, convert to 0-based for Python
        zero_indices.append(zi_arr - 1)

    # 4. Load or construct X_missing (Observed data per view)
    if 'X_missing' in mat:
        X_missing = extract_matrices(mat['X_missing'])
        for v in range(view_num):
            nv = n - len(zero_indices[v])
            if X_missing[v].ndim == 1:
                if X_missing[v].shape[0] == nv:
                    X_missing[v] = X_missing[v].reshape(1, nv)
                else:
                    X_missing[v] = X_missing[v].reshape(-1, 1)
                    
            # Ensure orientation (dv, nv)
            if X_missing[v].shape[0] == nv and X_missing[v].shape[1] != nv:
                X_missing[v] = X_missing[v].T
            elif X_missing[v].shape[0] != nv and X_missing[v].shape[1] == nv:
                pass # Already correct
            else:
                # Fallback: compare with X[v] feature dimension
                dv = X[v].shape[0]
                if X_missing[v].shape[0] != dv and X_missing[v].shape[1] == dv:
                    X_missing[v] = X_missing[v].T
    else:
        # Construct X_missing if not explicitly saved
        X_missing = []
        for v in range(view_num):
            mask = np.ones(n, dtype=bool)
            mask[zero_indices[v]] = False
            X_missing.append(X[v][:, mask])

    # 5. Construct XF (Concatenated complete features) and M (Indicator matrices)
    XF = np.vstack(X)
    M = []
    for v in range(view_num):
        M_v = np.eye(n)
        M_v = np.delete(M_v, zero_indices[v], axis=0)
        M.append(M_v)

    return X_missing, XF, M, zero_indices, labels, class_num, n
import os
from typing import List, Tuple

import numpy as np
import scipy.io as sio


def load_dataset(filepath: str):
    """
    Load the original TERUIMUFS-style .mat dataset.

    This preserves the original TERUIMUFS data contract:
    - X
    - labels
    - missing_num
    - one_indices
    - XF
    - W
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    mat = sio.loadmat(filepath)

    X_mat = mat["X"][0]

    X = []
    for x in X_mat:
        if hasattr(x, "toarray"):
            x = x.toarray()
        X.append(np.asarray(x, dtype=float))

    labels = np.asarray(mat["labels"]).flatten().astype(int)

    missing_num = np.asarray(mat["missing_num"]).flatten().astype(int).tolist()
    if not isinstance(missing_num, list):
        missing_num = [missing_num]

    one_indices = [
        np.asarray(arr).flatten().astype(int) - 1
        for arr in mat["one_indices"][0]
    ]

    view_num = len(X)
    instance_num = X[0].shape[1]
    class_num = len(np.unique(labels))

    XF = np.vstack(X)

    W = []
    for v in range(view_num):
        Wv = np.eye(instance_num)
        Wv = np.delete(Wv, one_indices[v], axis=0)
        W.append(Wv)

    return (
        X,
        labels,
        missing_num,
        one_indices,
        XF,
        W,
        instance_num,
        class_num,
    )
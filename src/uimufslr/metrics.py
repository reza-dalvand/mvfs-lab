import numpy as np
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

def compute_f(T: np.ndarray, H: np.ndarray) -> tuple:
    """Pairwise F-score, Precision, Recall for clustering."""
    cont = contingency_matrix(T, H)
    numI = np.sum(cont * (cont - 1)) / 2.0
    sum_rows = np.sum(cont, axis=1)
    numT = np.sum(sum_rows * (sum_rows - 1)) / 2.0
    sum_cols = np.sum(cont, axis=0)
    numH = np.sum(sum_cols * (sum_cols - 1)) / 2.0
    
    p = numI / numH if numH > 0 else 1.0
    r = numI / numT if numT > 0 else 1.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f, p, r

def evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Computes ACC, NMI, Purity, F-score, Precision, Recall, ARI.
    Faithfully reproduces EvaluationMetrics.m
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    cont = contingency_matrix(y_true, y_pred)
    
    # ACC via Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-cont)
    acc_val = cont[row_ind, col_ind].sum() / cont.sum()
    
    # NMI (max normalization matches MATLAB's MIhat)
    nmi_val = normalized_mutual_info_score(y_true, y_pred, average_method='max')
    
    # Purity
    pu_val = np.sum(np.max(cont, axis=0)) / np.sum(cont)
    
    # Pairwise metrics
    fs, pre, rec = compute_f(y_true, y_pred)
    
    # ARI
    ari_val = adjusted_rand_score(y_true, y_pred)
    
    return acc_val, nmi_val, pu_val, fs, pre, rec, ari_val
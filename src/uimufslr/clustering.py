from sklearn.cluster import KMeans
from .metrics import evaluation_metrics

def evaluate_feature_selection(Xsub, labels, class_num, random_state=None):
    """Evaluates clustering performance on selected features."""
    # n_jobs removed in newer scikit-learn versions
    kmeans = KMeans(n_clusters=class_num, n_init=25, max_iter=100, random_state=random_state)
    pred_labels = kmeans.fit_predict(Xsub)
    acc, nmi, pu, fs, pre, rec, ari = evaluation_metrics(labels, pred_labels)
    return {
        "acc": acc, "nmi": nmi, "purity": pu,
        "fscore": fs, "precision": pre, "recall": rec, "ari": ari
    }
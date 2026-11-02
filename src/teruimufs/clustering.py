from sklearn.cluster import KMeans

def litekmeans(X, k, MaxIter=100, Replicates=1, distance='sqeuclidean'):
    km = KMeans(n_clusters=k, n_init=Replicates, max_iter=MaxIter, random_state=None)
    return km.fit_predict(X)